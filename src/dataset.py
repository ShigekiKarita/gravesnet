import glob
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


def extract_raw_strokes(xml_path, removal_threshold=100.0):
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


def interpolate_strokes(raw_strokes, interval=0.01):
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


def prepare_stroke_line(raw_line, interval):
    xs = interpolate_strokes(raw_line, interval)
    es = create_endpoints(xs)

    # add lift-off seqs inter-strokes
    end = [r[-1][0] for r in raw_line][:-1]
    next = [r[0][0] for r in raw_line][1:]
    off = [int((n - e) / interval) for n, e in zip(next, end)]
    off += [int(numpy.mean(off))]
    xs = [numpy.concatenate((x, x[-1] * o), axis=1) for x, o in zip(xs, off)]
    es = [numpy.concatenate((e, e[-1] * o), axis=1) for e, o in zip(es, off)]
    return xs, es


def normalize_strokes(strokes):
    return (strokes - numpy.mean(strokes, 0)) / numpy.std(strokes, 0)


def parse_IAMxml(xml_path):
    strokes = extract_raw_strokes(xml_path)
    strokes = interpolate_strokes(strokes)
    endpoints = create_endpoints(strokes)
    endpoints = numpy.concatenate(endpoints)
    strokes = numpy.concatenate(strokes)
    strokes = normalize_strokes(strokes)
    return strokes, endpoints.reshape(len(endpoints), 1)


def parse_IAMtxt(txt_path, data_dir_path):
    result = []
    with open(txt_path, 'r') as f:
        line = f.readline().strip()
        while line:
            a = line[:3]
            b = line[:7]
            dir = "{}/{}/{}/".format(data_dir_path, a, b)
            files = glob.glob(dir + line + "*.xml")
    return result


def parse_IAMdataset(txt_path, data_dir_path):
    strokes_set = parse_IAMtxt(txt_path, data_dir_path)
    raw_strokes = []
    for line_strokes in strokes_set:
        for line in line_strokes:
            
            raw_strokes += extract_raw_strokes(line)
    # ignore, regard multiple lines as one line
    xs, es = prepare_stroke_line(raw_strokes, 0.01)
    es = numpy.concatenate(es).reshape(len(es), 1)
    xs = numpy.concatenate(xs)
    xs = normalize_strokes(xs)
    return xs, es
