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


@pytest.mark.parametrize('db', [('selected_ngram')], indirect=True)
def test_selected_ngram(db):

    db.selected_ngram(None, None, new=[1])


@pytest.mark.parametrize('db', [('selected_topic')], indirect=True)
def test_selected_topic(db):

    db.selected_topic(None, None, new=[0])


@pytest.mark.parametrize('db', [('rename_topic')], indirect=True)
def test_rename_topic(db):

    db.selected_topic(None, None, new=[0])
    db.input_topic_name.value = 'Topic Renamed'
    db.rename_topic(None)


@pytest.mark.parametrize('db', [('get_topic_prediction')], indirect=True)
def test_get_topic_prediction(db):

    db.predict['input'].value = """football season is over. so I'll have more time to learn about my new mac"""
    db.get_topic_prediction(None)


@pytest.mark.parametrize('db', [('recalculate_model')], indirect=True)
def test_recalculate_model(db):

    db.model_inputs['stop_words'].value = 'AX'
    db.recalculate_model(None)