#  Copyright (c) 2022 Dafne-Imaging Team
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QSizePolicy

from ..config import GlobalConfig, save_config


class WhatsNewDialog(QDialog):
    def __init__(self, date, link, title, excerpt, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        myLayout = QVBoxLayout(self)
        self.setLayout(myLayout)
        self.setWindowTitle(f"Dafne News")
        self.setWindowModality(Qt.ApplicationModal)
        date_label = QLabel(f'<b>{date}</b>')
        date_label.sizePolicy().setVerticalStretch(0)
        myLayout.addWidget(date_label)
        title_label = QLabel(f'<h2><a href="{link}">{title}</a></h2>')
        title_label.setOpenExternalLinks(True)
        title_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        title_label.sizePolicy().setVerticalStretch(0)
        title_label.setWordWrap(True)
        myLayout.addWidget(title_label)
        body_label = QLabel(excerpt)
        body_label.setWordWrap(True)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(body_label.sizePolicy().hasHeightForWidth())
        body_label.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignTop)
        body_label.setSizePolicy(sizePolicy)
        myLayout.addWidget(body_label)
        btn = QPushButton("Close")
        btn.clicked.connect(self.close)
        myLayout.addWidget(btn)
        self.resize(300,200)
        self.show()


def xml_timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')


def datetime_to_xml_timestamp(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S%z')


def check_for_updates():
    last_news_time = xml_timestamp_to_datetime(GlobalConfig['LAST_NEWS'])
    try:
        r = requests.get(GlobalConfig['NEWS_URL'])
    except requests.exceptions.ConnectionError:
        return []

    if r.status_code != 200:
        return []
    try:
        feed = ET.fromstring(r.text)
    except ET.ParseError:
        print("Error parsing news feed")
        return []

    parsed_uri = urlparse(GlobalConfig['NEWS_URL'])
    base_url = f'{parsed_uri.scheme}://{parsed_uri.netloc}'

    xml_ns = {'atom': 'http://www.w3.org/2005/Atom'}

    news_list = []
    newest_time = last_news_time
    for entry in feed.findall('atom:entry', xml_ns):
        link = entry.find('atom:link', xml_ns).attrib['href']
        # skip the index
        if link == '/blog/index/':
            continue

        news_time = xml_timestamp_to_datetime(entry.find('atom:updated', xml_ns).text)
        if news_time > last_news_time:
            news_list.append({'date': news_time.strftime('%Y-%m-%d'),
                              'link': base_url + link,
                              'title': entry.find('atom:title', xml_ns).text,
                              'excerpt': entry.find('atom:summary', xml_ns).text})
            if news_time > newest_time:
                newest_time = news_time

    GlobalConfig['LAST_NEWS'] = datetime_to_xml_timestamp(newest_time)
    save_config()
    return news_list


def show_news():
    for news in check_for_updates():
        d = WhatsNewDialog(**news)
        d.exec()


def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    show_news()