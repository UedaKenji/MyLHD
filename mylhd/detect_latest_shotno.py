from mylhd.anadata import KaisekiData
import os


dir_path = os.path.dirname(__file__)

def detect_latest_shotno(startno = None):

    if startno is None:
        shotno = get_shotno_from_file(filename=os.path.join(dir_path, "latest_shotno.txt"))

    for shotno in range(shotno, shotno+10000):
        try:
            _ = KaisekiData.retrieve_opendata(shotno=shotno, diag='shotinfo')
        except:
            break
    shotno -= 1

    write_latest_shotno(shotno, filename=os.path.join(dir_path, "latest_shotno.txt"))

    return shotno


def get_shotno_from_file(filename=os.path.join(dir_path, "latest_shotno.txt")):
    """Get the latest shot number from a file.
    Args:
        filename (str): Path to the file containing the latest shot number.
    Returns:
        int: Latest shot number.
    """
    with open(filename, "r") as f:
        shotno_str = f.read().strip()
        return int(shotno_str) if shotno_str.isdigit() else None

def write_latest_shotno(shotno, filename=os.path.join(dir_path, "latest_shotno.txt")):
    """Write the latest shot number to a file.
    Args:
        shotno (int): Latest shot number.
        filename (str): Path to the file to write the shot number.
    """
    with open(filename, "w") as f:
        f.write(str(shotno))
    


if __name__ == "__main__":
    latest_shotno = detect_latest_shotno()
    print(f"Latest shot number detected: {latest_shotno}")

