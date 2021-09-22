
def get_groundtruth(
    found_words,
    targets,
    groundtruth,
    time_tolerance_ms=1500,
):
    detections = []
    for target in targets:
        gt_target_times = [t for f, t in groundtruth if f == target]
        print("gt target occurences", len(gt_target_times))
        found_target_times = [t for f, t, d in found_words if f == target]
        print("num found targets", len(found_target_times))

        #false negatives
        for time in groundtruth:
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for found_time in found_target_times:
                if found_time > latest_time:
                    break
                if found_time < earliest_time:
                    continue
                potential_match = True
            if not potential_match:
                #false negative
                detections.append([target, time, 1, 'fn'])

        # true positives / false positives
        for time in found_target_times:
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms

            potential_match = False
            for gt_time in gt_target_times:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                potential_match = True
            if potential_match: #true positive
                detections.append([target, time, d, 'tp'])
            else: #false positive
                detections.append([target, time, d, 'fp'])

        return detections
