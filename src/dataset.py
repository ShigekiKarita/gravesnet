import numpy
import scipy.interpolate

try:
    from lxml import parse
except ImportError:
    from xml.etree.ElementTree import parse


def make_tuple(point):
    x = int(point.attrib["x"])
    y = int(point.attrib["y"])
    t = float(point.attrib["time"])
    return numpy.array([x, y]), t


def extract_raw_strokes(xml_path, removal_threshold):
    root = parse(xml_path).getroot()
    stroke_set = root.find("StrokeSet")
    result = []
    for stroke in stroke_set.iter("Stroke"):
        xys = []
        ts = []
        first = stroke.find("Point")
        p_xy, _ = make_tuple(first)
        if len(stroke.findall("Point")) < 2: # maybe noise
            continue
        for point in stroke.iter("Point"):
            xy, t = make_tuple(point)
            if numpy.linalg.norm(xy - p_xy) > removal_threshold: # maybe noise
                continue
            p_xy = xy
            xys.append(xy)
            ts.append(t)
        result.append((numpy.array(ts), numpy.array(xys).transpose()))
    return result


def interpolate_strokes(raw_strokes, interval):
    result = []
    for t, xy in raw_strokes:
        f = scipy.interpolate.interp1d(t, xy)
        r = [numpy.float32(f(t)) for t in numpy.arange(t[0], t[-1], interval)]
        result.append(numpy.array(r))
    return result


def create_endpoints(strokes):
    result = []
    for s in strokes:
        r = numpy.zeros(len(s), dtype=numpy.float32)
        r[-1] = 1.0
        result.append(r)
    return result


def normalize_strokes(strokes):
    return (strokes - numpy.mean(strokes, 0)) / numpy.std(strokes, 0)


def parse_IAMxml(xml_path):
    strokes = extract_raw_strokes(xml_path, 100.0)
    strokes = interpolate_strokes(strokes, 0.01)
    endpoints = create_endpoints(strokes)
    endpoints = numpy.concatenate(endpoints)
    strokes = numpy.concatenate(strokes)
    strokes = normalize_strokes(strokes)
    return strokes, endpoints

