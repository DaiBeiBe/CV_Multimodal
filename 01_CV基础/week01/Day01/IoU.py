def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = intersection_width * intersection_height
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - intersection_area

    iou = intersection_area / union_area

    return iou

if __name__ == "__main__":
    box1 = [10, 10, 50, 50]
    box2 = [10, 10, 50, 50] 
    print(calculate_iou(box1, box2))
    box1 = [10, 10, 20, 20]
    box2 = [30, 30, 40, 40]
    print(calculate_iou(box1, box2))
    box1 = [10, 10, 50, 50]
    box2 = [30, 30, 70, 70]  
    print(calculate_iou(box1, box2))