def get_groundtruth(
    found_words, targets, groundtruth, time_tolerance_ms=1500,
):
    detections = []
    for target in targets:
        gt_target_times = [t for k, t in groundtruth if k == target]
        print("gt target occurences", len(gt_target_times))
        found_target_times = [found for found in found_words if found[0] == target]
        print("num found targets", len(found_target_times))

        # false negatives
        for time in gt_target_times:
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for _, found_time, _ in found_target_times:
                if found_time > latest_time:
                    break
                if found_time < earliest_time:
                    continue
                potential_match = True
            if not potential_match:
                # false negative
                detections.append(dict(keyword=target, time_ms=time, groundtruth="fn"))

        # true positives / false positives
        for _, time, confidence in found_target_times:
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms

            potential_match = False
            for gt_time in gt_target_times:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                potential_match = True

            # these tp/fp classifications are wrt the (minimum) detection confidence threshold
            # and do not reflect a higher user threshold in the visualizer
            if potential_match:  # true positive
                detections.append(
                    dict(
                        keyword=target,
                        time_ms=time,
                        confidence=confidence,
                        groundtruth="tp",
                    )
                )
            else:  # false positive
                detections.append(
                    dict(
                        keyword=target,
                        time_ms=time,
                        confidence=confidence,
                        groundtruth="fp",
                    )
                )

        return detections


def tpr_fpr(
    keyword,
    thresh,
    found_words,
    gt_target_times_ms,
    duration_s,
    time_tolerance_ms,
    num_nontarget_words=None,
):
    found_target_times = [t for f, t in found_words if f == keyword]

    # find false negatives
    false_negatives = 0
    for time_ms in gt_target_times_ms:
        latest_time = time_ms + time_tolerance_ms
        earliest_time = time_ms - time_tolerance_ms
        potential_match = False
        for found_time in found_target_times:
            if found_time > latest_time:
                break
            if found_time < earliest_time:
                continue
            potential_match = True
        if not potential_match:
            false_negatives += 1

    # find true/false positives
    false_positives = 0  # no groundtruth match for model-found word
    true_positives = 0
    for word, time in found_words:
        if word == keyword:
            # highlight spurious words
            latest_time = time + time_tolerance_ms
            earliest_time = time - time_tolerance_ms
            potential_match = False
            for gt_time in gt_target_times_ms:
                if gt_time > latest_time:
                    break
                if gt_time < earliest_time:
                    continue
                potential_match = True
            if not potential_match:
                false_positives += 1
            else:
                true_positives += 1
    if true_positives > len(gt_target_times_ms):
        print("WARNING: weird timing issue")
        true_positives = len(gt_target_times_ms)
        # if thresh is low, mult dets map to single gt (above suppression_ms)
        # single_target_recognize_commands already uses suppression_ms
        # raise suppression value?

    tpr = true_positives / len(gt_target_times_ms)
    false_rejections_per_instance = false_negatives / len(gt_target_times_ms)
    false_positives = len(found_target_times) - true_positives

    fah = false_positives / duration_s * 3600  # sec/hr
    result = dict(
        keyword=keyword,
        tpr=tpr,
        thresh=thresh,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        false_rejections_per_instance=false_rejections_per_instance,
        false_accepts_per_hour=fah,
        groundtruth_positives=len(gt_target_times_ms),
    )
    # false_accepts_per_seconds = false_positives / (duration_s / (3600))

    # TODO(mmaz) does this hold true across multiple time_tolerance_ms values?
    # fpr = false_positives / gt_negatives == false_positives / (false_positives + true_negatives)
    if num_nontarget_words is not None:
        result["fpr"] = false_positives / num_nontarget_words

    return result
