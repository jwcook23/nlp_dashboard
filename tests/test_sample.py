import pytest
from bokeh.plotting import output_file, show

from nlp_newsgroups.dashboard import dashboard

@pytest.fixture(scope='module')
def db(request):

    page = dashboard(server=False, standalone=False)

    yield page

    file_name = request.param

    output_file(f'tests/{file_name}.html')
    show(page.layout)


@pytest.mark.parametrize('db', [('ngram')], indirect=True)
def test_ngram(db):

    db.selected_ngram(attr=None, old=None, new=[0])


@pytest.mark.parametrize('db', [('topic_lda')], indirect=True)
def test_topic_lda(db):

    db.selected_topic(attr=None, old=None, new=[0])

# import subprocess
# from pathlib import Path

# import pytest
# from selenium import webdriver
# from selenium.webdriver.common.by import By


# @pytest.fixture(scope='module')
# def browser():

#     cwd = Path().absolute()
#     path_bokeh = Path(cwd,'env','Scripts','bokeh')
#     path_dashboard = Path(cwd,'nlp_newsgroups','dashboard.py')
    
#     process = subprocess.Popen([path_bokeh, 'serve', path_dashboard])

#     driver = webdriver.Chrome()

#     driver.get(f"http://localhost:5006/dashboard")

#     yield driver

#     driver.close()
#     process.kill()


# def test_ngram(browser):

#     shadow = browser.execute_script('''return document.querySelector(".bk-Column").shadowRoot''')
#     element = shadow.find_element(By.CLASS_NAME, "bk-Figure")
#     element.click()