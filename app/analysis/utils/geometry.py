# utils/geometry.py

def increase_bounding_box(box, video_w, video_h):
    new_box = {}
    new_box['x'] = int(max(0, box['x'] - box['width'] * 0.125))
    new_box['y'] = int(max(0, box['y'] - box['height'] * 0.125))
    new_box['width'] = int(min(video_w - new_box['x'], box['width'] * 1.25))
    new_box['height'] = int(min(video_h - new_box['y'], box['height'] * 1.25))
    return new_box
