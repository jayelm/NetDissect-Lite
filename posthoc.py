"""
Traverse through the HTML, post-hoc
"""

from bs4 import BeautifulSoup
import loader.data_loader.formula as F
import os


def load_html(html_fname):
    with open(html_fname, 'r') as f:
        html = f.read()
    return BeautifulSoup(html, 'html.parser')


def units(soup, layername=None, yield_soup=False):
    """
    Iterate through units given hhtml
    """
    unit_soup = soup.find_all('div', 'unit')

    for us in unit_soup:
        unit = us.find('span', 'unitnum').text.strip().split('unit ')[1]
        iou = us.find('span', 'iou').text.strip().split('IoU ')[1]
        label = us.find('div', 'unitlabel').text.strip()
        # consistency
        label_str, consistency = label.split(' (consistency: ')
        consistency = consistency.split(')')[0]

        label = F.parse(label_str)

        record = {
            'layer': layername,
            'label_str': label_str,
            'label': label,
            'consistency': float(consistency),
            'unit': int(unit),
            'iou': float(iou)
        }
        if yield_soup:
            yield record, us
        else:
            yield record
