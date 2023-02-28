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

    db.selected_ngram(None, None, new=[1])


@pytest.mark.parametrize('db', [('topic_lda')], indirect=True)
def test_topic_lda(db):

    db.selected_topic(None, None, new=[0])


@pytest.mark.parametrize('db', [('add_stopword')], indirect=True)
def test_add_stopword(db):

    db.add_stopword(None, None, new='AX')