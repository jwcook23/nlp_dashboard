
from bokeh.server.server import Server
from selenium import webdriver

from nlp_newsgroups.dashboard import dashboard

def _bkapp(doc):

    db = dashboard(server=False, standalone=False)

    doc.add_root(db.layout)


def get_server():

    server = Server({'/': _bkapp})
    server.start()

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

    driver = webdriver.Chrome()

    driver.get(f"http://localhost:5006/dashboard")

    return driver


def test_ngram():

    # https://stackoverflow.com/questions/49823206/how-to-find-a-button-with-selenium-webdriver-in-a-bokeh-document
    driver = get_server()