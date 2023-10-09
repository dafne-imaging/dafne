#  Copyright (c) 2022 Dafne-Imaging Team
from datetime import datetime
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QSizePolicy

from ..config import GlobalConfig, save_config
from ..utils.ThreadHelpers import separate_thread_decorator

MAX_NEWS_ITEMS = 3
BLOG_INDEX = '/blog/index/'
BLOG_ADDRESS = '/blog/'


class WhatsNewDialog(QDialog):
    def __init__(self, news_list, index_address, *args, **kwargs):
        QDialog.__init__(self, *args, **kwargs)
        my_layout = QVBoxLayout(self)
        self.setLayout(my_layout)
        self.setWindowTitle(f"Dafne News")
        self.setWindowModality(Qt.ApplicationModal)

        for news in news_list[:MAX_NEWS_ITEMS]:
            title_label = QLabel(f'<h2><a href="{news["link"]}">{news["title"]}</a></h2>')
            title_label.setOpenExternalLinks(True)
            title_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
            title_label.sizePolicy().setVerticalStretch(0)
            title_label.setWordWrap(True)
            my_layout.addWidget(title_label)
            date_label = QLabel(f'<b>{news["date"]}</b>')
            date_label.sizePolicy().setVerticalStretch(0)
            my_layout.addWidget(date_label)
            body_label = QLabel(news["excerpt"])
            body_label.setWordWrap(True)
            size_policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            size_policy.setHorizontalStretch(0)
            size_policy.setVerticalStretch(1)
            size_policy.setHeightForWidth(body_label.sizePolicy().hasHeightForWidth())
            body_label.setAlignment(Qt.AlignLeading | Qt.AlignLeft | Qt.AlignTop)
            body_label.setSizePolicy(size_policy)
            my_layout.addWidget(body_label)

        more_news_label = QLabel(f'<a href="{index_address}">All news...</a>')
        more_news_label.setOpenExternalLinks(True)
        more_news_label.setTextInteractionFlags(Qt.LinksAccessibleByMouse)
        more_news_label.sizePolicy().setVerticalStretch(0)
        my_layout.addWidget(more_news_label)

        n_news = min(MAX_NEWS_ITEMS, len(news_list))

        btn = QPushButton("Close")
        btn.clicked.connect(self.close)
        my_layout.addWidget(btn)
        self.resize(300, 110 * n_news + 60)
        self.show()


def xml_timestamp_to_datetime(timestamp):
    return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S%z')


def datetime_to_xml_timestamp(dt):
    return dt.strftime('%Y-%m-%dT%H:%M:%S%z')


def check_for_updates():
    last_news_time = xml_timestamp_to_datetime(GlobalConfig['LAST_NEWS'])
    # last_news_time = xml_timestamp_to_datetime('2010-11-10T00:00:00+00:00')
    try:
        r = requests.get(GlobalConfig['NEWS_URL'], timeout=(1, None))
    except requests.exceptions.ConnectionError:
        return [], []
    except requests.exceptions.Timeout:
        return [], []

    if r.status_code != 200:
        return [], []
    try:
        feed = ET.fromstring(r.text)
    except ET.ParseError:
        print("Error parsing news feed")
        return [], []

    parsed_uri = urlparse(GlobalConfig['NEWS_URL'])
    base_url = f'{parsed_uri.scheme}://{parsed_uri.netloc}'

    xml_ns = {'atom': 'http://www.w3.org/2005/Atom'}

    news_list = []
    newest_time = last_news_time
    for entry in feed.findall('atom:entry', xml_ns):
        link = entry.find('atom:link', xml_ns).attrib['href']
        # skip the index
        if link == BLOG_INDEX:
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
    news_list.sort(key=lambda x: x['date'], reverse=True)
    save_config()
    return news_list, base_url + BLOG_ADDRESS


class NewsChecker(QObject):
    news_ready = pyqtSignal(list, str)

    def __init__(self, *args, **kwargs):
        QObject.__init__(self, *args, **kwargs)

    @separate_thread_decorator
    def check_news(self):
        news_list, index_address = check_for_updates()
        if news_list:
            self.news_ready.emit(news_list, index_address)


def show_news():
    news_list, index_address = check_for_updates()
    if news_list:
        d = WhatsNewDialog(news_list, index_address)
        d.exec()


def main():
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    show_news()
