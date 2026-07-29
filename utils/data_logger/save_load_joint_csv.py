import csv
import numpy as np


def save_trajectory_csv(filename, Q, time_len):
    """
    Save one trajectory to CSV. In the format of:
    Time length, time_len
    step, q1, q2, ..., q15

    Parameters
    ----------
    filename : str or Path
        Output CSV file path
    Q : np.ndarray
        Shape (N, D), trajectory data (N steps, D joints)
    time_len : float
        Total trajectory duration (seconds)
    """
    N, D = Q.shape
    # assert D == 15, f"Expected 15 DoF, got {D}"

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)

        # First line: time length
        writer.writerow(["time_len", time_len])

        # Header line
        header = ["step"] + [f"q{i+1}" for i in range(D)]
        writer.writerow(header)

        # Data rows
        for step, row in enumerate(Q):
            writer.writerow([step] + list(row))


def load_trajectory_csv(filename):
    """
    Load trajectory and time_len back from CSV.

    Returns
    -------
    Q : np.ndarray of shape (N, D)
    time_len : float
    """
    with open(filename, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # first row is time_len
    time_len = float(rows[0][1])

    # remaining rows: header + data
    header = rows[1]
    data_rows = rows[2:]

    # Extract number of joints from header (header is ["step", "q1", "q2", ...])
    num_joints = len(header) - 1

    if len(data_rows) == 0:
        # Handle empty trajectory case - preserve the number of joints
        Q = np.empty((0, num_joints))
    else:
        Q = np.array([[float(x) for x in row[1:]] for row in data_rows])

    return Q, time_len
