import random
import string


def random_name(length=8):
    letters = string.ascii_lowercase + string.digits
    return "".join(random.choice(letters) for _ in range(length))
