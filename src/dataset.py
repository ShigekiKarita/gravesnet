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
        p_xy, p_t = make_tuple(first)
        for point in stroke.iter("Point"):
            xy, t = make_tuple(point)
            if numpy.linalg.norm(xy - p_xy) > removal_threshold or p_t == t: # maybe noise
                continue
            p_xy = xy
            p_t = t
            xys.append(xy)
            ts.append(t)
        if len(ts) < 2: # unable to interpolate
            continue
        result.append((numpy.array(ts), numpy.array(xys).transpose()))
    return result


def extract_raw_text(xml_path):
    root = parse(xml_path).getroot()
    transcription = root.find("Transcription")
    return [t.attrib["text"] for t in transcription.iter("TextLine")]


def interpolate_strokes(raw_strokes, interval=0.01):
    result = []
    for t, xy in raw_strokes:
        if t[-1] - t[0] < interval:
            continue
        try:
            f = scipy.interpolate.interp1d(t, xy)
            r = []
            for t in numpy.arange(t[0], t[-2], interval):
                r.append(f(t))
        except:
            print("error -> continue!!!")
            continue
        if len(r) > 2:
            result.append(numpy.array(r))
    return result


def create_endpoints(strokes):
    result = []
    for s in strokes:
        r = numpy.zeros(len(s))
        r[-1] = 1.0
        result.append(r)
    return result


def add_lift_offs(xs, off):
    return [numpy.concatenate((x, [x[-1]] * o)) for x, o in zip(xs, off)]


def prepare_stroke_line(raw_line, interval):
    xs = interpolate_strokes(raw_line, interval)
    es = create_endpoints(xs)

    # add lift-off seqs inter-strokes
    off = []
    if len(raw_line) == 1:
        off = [int(numpy.random.uniform(20.0, 50.0))]
    else:
        end = [r[0][-1] for r in raw_line][:-1]
        next = [r[0][0] for r in raw_line][1:]
        off = [int(abs(n - e) / interval) + 1 for n, e in zip(next, end)]
        off += [int(numpy.mean(off)) + 1]
    xs = add_lift_offs(xs, off)
    es = add_lift_offs(es, off)
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


def parse_dir(label):
    a = label[:3]
    b = label[:7]
    return "/{}/{}/".format(a, b)


def parse_IAM_files(txt_path, root, dir, func):
    result = []
    with open(txt_path, 'r') as f:
        line = f.readline()
        while line:
            label = line.strip()
            d = parse_dir(label)
            x = root + dir + d + label
            result.append(func(x))
            line = f.readline()
    return result


def parse_IAM_stroke_file(txt_path, root):
    f = lambda x: glob.glob(x + "*.xml")
    return parse_IAM_files(txt_path, root, "lineStrokes", f)


def parse_IAM_ascii_file(txt_path, root):
    f = lambda x: x + ".txt"
    return parse_IAM_files(txt_path, root, "ascii", f)


def extract_ascii(txt_path):
    result = []
    with open(txt_path, 'r') as f:
        line = f.readline()
        while not line.startswith("CSR:"):
            line = f.readline()
        f.readline()
        line = f.readline().strip()
        while line:
            result.append(line)
            line = f.readline().strip()
        return result


def parse_IAMdataset_strokes(txt_path, data_dir_path):
    numpy.seterr(divide="raise")
    strokes_set = parse_IAM_stroke_file(txt_path, data_dir_path)
    print("processed file list")
    xs, es, ts = [], [], []
    for line_strokes in strokes_set:
        for line_file_path in line_strokes:
            print(line_file_path)
            raw_strokes = extract_raw_strokes(line_file_path)
            x, e = prepare_stroke_line(raw_strokes, 0.01)
            xs.append(x)
            es.append(e)
    return xs, es


def one_hot_vector(line: str):
    

def parse_IAMdataset_ascii(txt_path, data_dir_path):
    file_set = parse_IAM_ascii_file(txt_path, data_dir_path)
    print("processed file list")
    ts = []
    for file in file_set:
        for line in extract_ascii(file):
           ts.append(one_hot_vector(line))
    return ts
