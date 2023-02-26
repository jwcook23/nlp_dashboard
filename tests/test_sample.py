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

    db.selected_ngram(attr=None, old=None, new=[1])


@pytest.mark.parametrize('db', [('topic_lda')], indirect=True)
def test_topic_lda(db):

    db.selected_topic(attr=None, old=None, new=[0])
