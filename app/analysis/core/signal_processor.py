# analysis/core/signal_processor.py

import numpy as np
import scipy.signal as signal
import scipy.interpolate as sp_interp

class SignalAnalyzer:
    """
    Handles signal processing, including:
      - optional filtering
      - upsampling to 60 FPS
      - peak finding
      - computing cycle stats
    """

    def analyze(self, task, display_landmarks, normalization_factor, fps, start_time, end_time):
        """
        1) get raw signal from the task
        2) normalize & upsample
        3) run peakFinder
        4) compute final stats (linePlot, velocityPlot, peaks, valleys, radar, etc.)
        5) return dict with old keys so front-end remains compatible
        """
        # 1) get raw signal
        raw_signal = task.get_signal(display_landmarks)
        signal_array = np.array(raw_signal) / (normalization_factor if normalization_factor else 1.0)
        duration = end_time - start_time

        # 2) upsample to 60 FPS
        upsample_fps = 60
        n_samples = int(duration * upsample_fps)
        if n_samples < 2:
            n_samples = len(signal_array)
        up_sample_signal = signal.resample(signal_array, n_samples)

        # 3) run peak finder
        distance, velocity, peaks = self.run_peak_finding(up_sample_signal)

        # Build time array for the distance & velocity
        line_time = []
        sizeOfDist = len(distance)
        for i in range(sizeOfDist):
            t = (i / sizeOfDist) * duration + start_time
            line_time.append(t)

        # 4) Reproduce old keys: linePlot, velocityPlot, rawData, peaks, valleys, etc.

        # Prepare arrays for storing peak/valley Y-values + times
        line_peaks = []
        line_peaks_time = []
        line_valleys = []        # used if you want to store 'middle' valley
        line_valleys_time = []
        line_valleys_start = []
        line_valleys_start_time = []
        line_valleys_end = []
        line_valleys_end_time = []

        # For each peak, gather the distance values at openingValleyIndex, peakIndex, closingValleyIndex
        for pk in peaks:
            # pk structure from your 'peakFinder': 
            # pk['peakIndex'], pk['openingValleyIndex'], pk['closingValleyIndex'], etc.
            pidx = pk['peakIndex']
            open_val_idx = pk['openingValleyIndex']
            close_val_idx = pk['closingValleyIndex']

            # Store peak data
            line_peaks.append(distance[pidx])
            line_peaks_time.append((pidx / sizeOfDist) * duration + start_time)

            # Opening valley
            line_valleys_start.append(distance[open_val_idx])
            line_valleys_start_time.append((open_val_idx / sizeOfDist) * duration + start_time)

            # Closing valley
            line_valleys_end.append(distance[close_val_idx])
            line_valleys_end_time.append((close_val_idx / sizeOfDist) * duration + start_time)

            # If you want a 'middle' valley concept, you can add it here or skip
            # line_valleys.append(...)
            # line_valleys_time.append(...)

        # 5) Example calculations for amplitude, velocity, etc. 
        #    (This is the style from your old get_output(...) code.)

        amplitude = []
        speed = []
        peakTime = []
        for i, pk in enumerate(peaks):
            open_val = pk['openingValleyIndex']
            close_val = pk['closingValleyIndex']
            peak_idx = pk['peakIndex']

            # baseline between opening & closing valley
            x1 = open_val
            x2 = close_val
            y1 = distance[x1]
            y2 = distance[x2]
            y_peak = distance[peak_idx]

            f = sp_interp.interp1d([x1, x2], [y1, y2], fill_value="extrapolate")
            baseline_at_peak = f(peak_idx)

            amp = y_peak - baseline_at_peak
            amplitude.append(amp)

            cycle_frames = (close_val - open_val)
            cycle_time = cycle_frames * (1 / upsample_fps)
            if cycle_time <= 0:
                cycle_time = 1e-6

            spd = amp / cycle_time
            speed.append(spd)

            # convert peak index to absolute time
            ptime = peak_idx * (1 / upsample_fps)
            peakTime.append(ptime)

        # Stats for radarTable example
        meanAmplitude = np.mean(amplitude) if amplitude else 0
        stdAmplitude  = np.std(amplitude)  if amplitude else 0
        meanSpeed     = np.mean(speed)     if speed else 0
        stdSpeed      = np.std(speed)      if speed else 0

        # dummy computations for demonstration
        amplitudeDecay = 1
        velocityDecay  = 1
        rateDecay      = 1

        # e.g. number of peaks / total time
        total_time = duration if duration > 0 else 1
        rate = len(peaks) / total_time if total_time else 0

        # 6) Build final dictionary 
        result = {
            "linePlot": {
                "data": distance.tolist(),
                "time": line_time
            },
            "velocityPlot": {
                "data": velocity.tolist(),
                "time": line_time
            },
            "rawData": {
                "data": up_sample_signal.tolist(),
                "time": line_time
            },
            "peaks": {
                "data": line_peaks,
                "time": line_peaks_time
            },
            "valleys": {
                # If your old code didn't store a "middle" valley, you can keep empty or remove this key
                "data": line_valleys,       
                "time": line_valleys_time
            },
            "valleys_start": {
                "data": line_valleys_start,
                "time": line_valleys_start_time
            },
            "valleys_end": {
                "data": line_valleys_end,
                "time": line_valleys_end_time
            },
            "radar": {
                # replicate your old "radar" arrays if the front-end expects them
                "A": [2.5, 3.8, 5.9, 4.2, 8.6, 4.9, 6.9, 9, 8.4, 7.3],
                "B": [3.8, 7.1, 7.4, 5.9, 4.2, 3.1, 6.5, 6.4, 3, 7],
                "labels": [
                    "Mean Frequency", "Mean cycle amplitude", "CV cycle amplitude",
                    "Mean cycle rms velocity","CV cycle rms velocity",
                    "Mean cycle duration", "CV cycle duration", 
                    "Range cycle duration","Amplitude decay","Velocity decay"
                ]
            },
            "radarTable": {
                "MeanAmplitude": meanAmplitude,
                "StdAmplitude": stdAmplitude,
                "MeanSpeed": meanSpeed,
                "StdSpeed": stdSpeed,
                # placeholders:
                "amplitudeDecay": amplitudeDecay,
                "velocityDecay": velocityDecay,
                "rateDecay": rateDecay,
                "rate": rate,
                # etc. add more if front-end expects them
            }
        }

        return result

    def run_peak_finding(self, signal_array):
        """
        Does the actual peakFinding from your old code:
        e.g. lowpass filter => velocity => positive/negative peaks => correctFullPeaks => ...
        Returns (distance, velocity, list_of_peaks)
        """
        distance, velocity, peaks, _, _ = peakFinder(
            signal_array,
            fs=60,
            minDistance=3,
            cutOffFrequency=7.5,
            prct=0.05
        )
        return distance, velocity, peaks


# ---------------  Peak-Finding Functions ---------------
import scipy.signal as signal

def compareNeighboursNegative(item1, item2, distance, minDistance=5):
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2
        # skip item2
        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['valleyIndex']] - distance[item2['peakIndex']]) < abs(
            distance[item1['valleyIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    return None


def compareNeighboursPositive(item1, item2, distance, minDistance=5):
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['peakIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['peakIndex']] - distance[item2['valleyIndex']]) < abs(
            distance[item1['peakIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    return None


def eliminateBadNeighboursNegative(indexVelocity, distance, minDistance=5):
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursNegative(indexVelocity[idx], indexVelocity[idx + 1], distance, minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected


def eliminateBadNeighboursPositive(indexVelocity, distance, minDistance=5):
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursPositive(indexVelocity[idx], indexVelocity[idx + 1], distance,
                                                    minDistance=minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected


def correctBasedonHeight(pos, distance, prct=0.125, minDistance=5):
    # eliminate any peaks that is smaller than 15% of the average height
    heightPeaks = []
    for item in pos:
        try:
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanHeightPeak = np.mean(heightPeaks)
    corrected = []
    for item in pos:
        try:
            if (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) > prct * meanHeightPeak:
                if abs(item['peakIndex'] - item['valleyIndex']) >= minDistance:
                    if (distance[item['peakIndex']] > distance[item['maxSpeedIndex']]) and (
                            distance[item['valleyIndex']] < distance[item['maxSpeedIndex']]):
                        corrected.append(item)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        except:
            pass

    return corrected


def correctBasedonVelocityNegative(pos, velocity, prct=0.125):
    # velocity[velocity>0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected


def correctBasedonVelocityPositive(pos, velocity, prct=0.125):
    velocity[velocity < 0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected


def correctFullPeaks(distance, pos, neg):
    # get the negatives
    closingVelocities = []
    for item in neg:
        closingVelocities.append(item['maxSpeedIndex'])

    openingVelocities = []
    for item in pos:
        openingVelocities.append(item['maxSpeedIndex'])

    peakCandidates = []
    for idx, closingVelocity in enumerate(closingVelocities):
        try:
            difference = np.array(openingVelocities) - closingVelocity
            difference[difference > 0] = 0

            posmin = np.argmax(difference[np.nonzero(difference)])

            absolutePeak = np.max(distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            absolutePeakIndex = pos[posmin]['maxSpeedIndex'] + np.argmax(
                distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            peakCandidate = {}

            peakCandidate['openingValleyIndex'] = pos[posmin]['valleyIndex']
            peakCandidate['openingPeakIndex'] = pos[posmin]['peakIndex']
            peakCandidate['openingMaxSpeedIndex'] = pos[posmin]['maxSpeedIndex']

            peakCandidate['closingValleyIndex'] = neg[idx]['valleyIndex']
            peakCandidate['closingPeakIndex'] = neg[idx]['peakIndex']
            peakCandidate['closingMaxSpeedIndex'] = neg[idx]['maxSpeedIndex']

            peakCandidate['peakIndex'] = absolutePeakIndex

            peakCandidates.append(peakCandidate)
        except:
            pass

    peakCandidatesCorrected = []
    idx = 0
    while idx < len(peakCandidates):

        peakCandidate = peakCandidates[idx]
        peak = peakCandidate['peakIndex']
        difference = [(peak - item['peakIndex']) for item in peakCandidates]
        if len(np.where(np.array(difference) == 0)[0]) == 1:
            peakCandidatesCorrected.append(peakCandidate)
            idx += 1
        else:
            item1 = peakCandidates[np.where(np.array(difference) == 0)[0][0]]
            item2 = peakCandidates[np.where(np.array(difference) == 0)[0][1]]
            peakCandidate = {}
            peakCandidate['openingValleyIndex'] = item1['openingValleyIndex']
            peakCandidate['openingPeakIndex'] = item1['openingPeakIndex']
            peakCandidate['openingMaxSpeedIndex'] = item1['openingMaxSpeedIndex']

            peakCandidate['closingValleyIndex'] = item2['closingValleyIndex']
            peakCandidate['closingPeakIndex'] = item2['closingPeakIndex']
            peakCandidate['closingMaxSpeedIndex'] = item2['closingMaxSpeedIndex']

            peakCandidate['peakIndex'] = item2['peakIndex']

            peakCandidatesCorrected.append(peakCandidate)
            idx += 2

    return peakCandidatesCorrected


def correctBasedonPeakSymmetry(peaks):
    peaksCorrected = []
    for peak in peaks:
        leftValley = peak['openingValleyIndex']
        centerPeak = peak['peakIndex']
        rightValley = peak['closingValleyIndex']

        ratio = (centerPeak - leftValley) / (rightValley - centerPeak)
        if 0.25 <= ratio <= 4:
            peaksCorrected.append(peak)

    return peaksCorrected


def peakFinder(rawSignal, fs=30, minDistance=5, cutOffFrequency=5, prct=0.125):
    indexPositiveVelocity = []
    indexNegativeVelocity = []

    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='lowpass', analog=False)

    distance = signal.filtfilt(b, a, rawSignal)  # signal.savgol_filter(rawDistance[0], 5, 3, deriv=0)
    velocity = signal.savgol_filter(distance, 5, 3, deriv=1) / (1 / fs)
    ##approx mean frequency
    acorr = np.convolve(rawSignal, rawSignal)
    t0 = ((1 / fs) * np.argmax(acorr))
    sep = 0.5 * (t0) if (0.5 * t0 > 1) else 1

    deriv = velocity.copy()
    deriv[deriv < 0] = 0
    deriv = deriv ** 2

    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksPositive = deriv[peaks]
    selectedPeaksPositive = peaks[heightPeaksPositive > prct * np.mean(heightPeaksPositive)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksPositive):
        idxValley = peak - 1
        if idxValley >= 0:
            while deriv[idxValley] != 0:
                if idxValley <= 0:
                    idxValley = np.nan
                    break

                idxValley -= 1

        idxPeak = peak + 1
        if idxPeak < len(deriv):
            while deriv[idxPeak] != 0:
                if idxPeak >= len(deriv) - 1:
                    idxPeak = np.nan
                    break

                idxPeak += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            positiveVelocity = {}
            positiveVelocity['maxSpeedIndex'] = peak
            positiveVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            positiveVelocity['peakIndex'] = idxPeak
            positiveVelocity['valleyIndex'] = idxValley
            indexPositiveVelocity.append(positiveVelocity)

    deriv = velocity.copy()
    deriv[deriv > 0] = 0
    deriv = deriv ** 2
    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksNegative = deriv[peaks]
    selectedPeaksNegative = peaks[heightPeaksNegative > prct * np.mean(heightPeaksNegative)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksNegative):

        idxPeak = peak - 1
        if idxPeak >= 0:
            while deriv[idxPeak] != 0:
                if idxPeak <= 0:
                    idxPeak = np.nan
                    break

                idxPeak -= 1

        idxValley = peak + 1
        if idxValley < len(deriv):
            while deriv[idxValley] != 0:
                if idxValley >= len(deriv) - 1:
                    idxValley = np.nan
                    break

                idxValley += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            negativeVelocity = {}
            negativeVelocity['maxSpeedIndex'] = peak
            negativeVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            negativeVelocity['peakIndex'] = idxPeak
            negativeVelocity['valleyIndex'] = idxValley
            indexNegativeVelocity.append(negativeVelocity)

            # euristics to remove bad peaks
    # # first, remove peaks that are too close to each other
    # indexPositiveVelocityCorrected = correctPeaksPositive(indexPositiveVelocity)    
    # indexNegativeVelocityCorrected = correctPeaksNegative(indexNegativeVelocity)
    # #then, remove peaks that are too small
    # indexPositiveVelocityCorrected = correctBasedonHeight(indexPositiveVelocityCorrected, distance)
    # indexNegativeVelocityCorrected = correctBasedonHeight(indexNegativeVelocityCorrected, distance)

    # remove bad peaks
    # 1- eliminate bad neighbours
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    # 2-eliminate bad peaks based on height
    indexPositiveVelocity = correctBasedonHeight(indexPositiveVelocity, distance)
    # 3-eliminate bad peaks based on velocity
    indexPositiveVelocity = correctBasedonVelocityPositive(indexPositiveVelocity, velocity.copy())

    # 1- eliminate bad neighbours
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    # 2-eliminate bad peaks based on height
    indexNegativeVelocity = correctBasedonHeight(indexNegativeVelocity, distance)
    # 3-eliminate bad peaks based on velocity
    indexNegativeVelocity = correctBasedonVelocityNegative(indexNegativeVelocity, velocity.copy())

    peaks = correctFullPeaks(distance, indexPositiveVelocity, indexNegativeVelocity)
    peaks = correctBasedonPeakSymmetry(peaks)

    return distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity
