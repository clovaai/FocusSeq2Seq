import re


tags = ['<t>', '</t>']


def read_split(path, strip_tag=True):
    lines = []

    def strip_tag(x):
        return x not in tags

    with open(path) as f:
        for line in f:
            line = line.strip().split()
            line = list(filter(strip_tag, line))
            lines.append(line)
    return lines


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def remove_tags(words):
    words = [word for word in words if word not in tags]
    return words


def split_sentences(words):
    # First, divide decoded output into sentences
    sents = []
    while len(words) > 0:
        try:
            fst_period_idx = words.index(".")
        except ValueError:  # there is text remaining that doesn't end in "."
            fst_period_idx = len(words)
        sent = words[:fst_period_idx + 1]  # sentence up to and including the period
        words = words[fst_period_idx + 1:]  # everything else
        sents.append(' '.join(sent))
    return sents


def split_tagged_sentences(article, sentence_start_tag='<t>', sentence_end_tag='</t>'):
    bare_sents = re.findall(r'%s (.+?) %s' % (sentence_start_tag, sentence_end_tag), article)
    return bare_sents
