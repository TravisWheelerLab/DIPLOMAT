

def _header_check(csv):
    with open(csv, "r") as csv_handle:
        first_lines = [csv_handle.readline().strip("\n").split(",") for i in range(3)]

        header_cols = len(first_lines[0])

        if(not all(header_cols == len(line) for line in first_lines)):
            return False

        last_header_line = first_lines[-1]
        last_line_exp = ["x", "y", "likelihood"] * (len(last_header_line) // 3)

        if(last_header_line != last_line_exp):
            return False

        return True


def _fix_paths(csvs, videos):
    csvs = csvs if(isinstance(csvs, (tuple, list))) else [csvs]
    videos = videos if(isinstance(videos, (tuple, list))) else [videos]

    if(len(csvs) == 1):
        csvs = csvs * len(videos)
    if(len(videos) == 1):
        videos = videos * len(csvs)

    if(len(videos) != len(csvs)):
        raise ValueError("Number of videos and csv files passes don't match!")

    return csvs, videos
