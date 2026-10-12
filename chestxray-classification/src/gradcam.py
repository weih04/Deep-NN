import torch
from captum.attr import LayerAttribution, LayerGradCam


def grad_cam(model, layer, images):
    """Grad-CAM heatmaps for the pneumonia logit, upsampled to the input size.

    The models output one logit, so no target index is needed. ReLU keeps only
    regions that push the prediction towards pneumonia. Each map is scaled to
    [0, 1] on its own, so maps show where a model looks, not how strongly
    relative to other images or models.
    """
    model.eval()
    attributions = LayerGradCam(model, layer).attribute(images, relu_attributions=True)
    heatmaps = LayerAttribution.interpolate(
        attributions, images.shape[-2:], interpolate_mode="bilinear"
    ).squeeze(1).detach()

    flat = heatmaps.flatten(1)
    low = flat.min(dim=1).values[:, None, None]
    high = flat.max(dim=1).values[:, None, None]
    # An all-zero map (no positive evidence) stays zero instead of dividing by 0.
    return (heatmaps - low) / (high - low).clamp_min(1e-12)


@torch.no_grad()
def pneumonia_probabilities(model, images):
    model.eval()
    return torch.sigmoid(model(images))


def chest_regions(size=64, border=4, central=(22, 42)):
    """Fixed rectangular masks: image border, central strip, lateral lung fields.

    The central columns cover the spine, mediastinum and heart. These are crude
    rectangles, not segmented lungs.
    """
    rows, cols = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    edge = (rows < border) | (rows >= size - border) | (cols < border) | (cols >= size - border)
    centre = ~edge & (cols >= central[0]) & (cols < central[1])
    return {
        "border": edge,
        "central_strip": centre,
        "lung_fields": ~edge & ~centre,
        "lower_half": rows >= size // 2,
    }


def region_shares(heatmaps, regions):
    """Share of each heatmap's total weight inside each region, averaged over images."""
    totals = heatmaps.sum(dim=(1, 2))
    return {
        name: ((heatmaps * mask).sum(dim=(1, 2)) / totals).mean().item()
        for name, mask in regions.items()
    }


def hot_region_brightness(heatmaps, images, threshold=0.5):
    """Mean intensity where heatmap >= threshold, minus the image mean; averaged."""
    differences = [
        image[heatmap >= threshold].mean() - image.mean()
        for heatmap, image in zip(heatmaps, images)
    ]
    return torch.stack(differences).mean().item()


def mean_correlation(first, second):
    """Pearson correlation between two models' maps for the same image, averaged."""
    pairs = torch.stack([first.flatten(1), second.flatten(1)], dim=1)
    return torch.stack([torch.corrcoef(pair)[0, 1] for pair in pairs]).mean().item()
