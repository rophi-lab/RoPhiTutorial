import re


def extract_pixel_from_text(text, h, w):
    coordinates = None

    if "<point" in text:
        matches = re.findall(r'(?:x(?:\d*)="([\d.]+)"\s*y(?:\d*)="([\d.]+)")', text)

        if len(matches) > 1:
            coordinates = [
                (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
                for x_val, y_val in matches
            ]
        else:
            coordinates = [
                (int(float(x_val) / 100 * w), int(float(y_val) / 100 * h))
                for x_val, y_val in matches
            ]
    return coordinates
