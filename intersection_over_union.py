import torch

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    # Convert midpoint format to corners format
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Use corners format as is
    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1] 
        box1_y1 = boxes_preds[..., 1:2] 
        box1_x2 = boxes_preds[..., 2:3] 
        box1_y2 = boxes_preds[..., 3:4] 

        box2_x1 = boxes_labels[..., 0:1] 
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3] 
        box2_y2 = boxes_labels[..., 3:4]

    else:
        raise ValueError("box_format must be either 'midpoint' or 'corners'")

    # Calculate the intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(min=0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Calculate the area of both boxes
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    # Calculate the intersection over union
    iou = intersection / (box1_area + box2_area - intersection + 1e-6)

    return iou


# Test case 1: Midpoint format, identical boxes
boxes_preds1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
boxes_labels1 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
iou1 = intersection_over_union(boxes_preds1, boxes_labels1, box_format="midpoint")
print(f"Test case 1 - IOU: {iou1}")

# Test case 2: Midpoint format, non-overlapping boxes
boxes_preds2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
boxes_labels2 = torch.tensor([[2.0, 2.0, 1.0, 1.0]])
iou2 = intersection_over_union(boxes_preds2, boxes_labels2, box_format="midpoint")
print(f"Test case 2 - IOU: {iou2}")

# Test case 3: Midpoint format, partially overlapping boxes
boxes_preds3 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
boxes_labels3 = torch.tensor([[0.75, 0.75, 1.0, 1.0]])
iou3 = intersection_over_union(boxes_preds3, boxes_labels3, box_format="midpoint")
print(f"Test case 3 - IOU: {iou3}")

# Test case 4: Corners format, identical boxes
boxes_preds4 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
boxes_labels4 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
iou4 = intersection_over_union(boxes_preds4, boxes_labels4, box_format="corners")
print(f"Test case 4 - IOU: {iou4}")

# Test case 5: Corners format, non-overlapping boxes
boxes_preds5 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
boxes_labels5 = torch.tensor([[2.0, 2.0, 3.0, 3.0]])
iou5 = intersection_over_union(boxes_preds5, boxes_labels5, box_format="corners")
print(f"Test case 5 - IOU: {iou5}")

# Test case 6: Corners format, partially overlapping boxes
boxes_preds6 = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
boxes_labels6 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
iou6 = intersection_over_union(boxes_preds6, boxes_labels6, box_format="corners")
print(f"Test case 6 - IOU: {iou6}")

# Test case 7: Mixed format, overlapping boxes
boxes_preds7 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
boxes_labels7 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])
iou7 = intersection_over_union(boxes_preds7, boxes_labels7, box_format="midpoint")
print(f"Test case 7 - IOU: {iou7}")

# Test case 8: Corners format, touching boxes
boxes_preds8 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
boxes_labels8 = torch.tensor([[1.0, 0.0, 2.0, 1.0]])
iou8 = intersection_over_union(boxes_preds8, boxes_labels8, box_format="corners")
print(f"Test case 8 - IOU: {iou8}")
