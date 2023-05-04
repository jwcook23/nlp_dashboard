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

    db.selected_source(None, None, row_source=[1], fig_name='ngram')


@pytest.mark.parametrize('db', [('selected_topic')], indirect=True)
def test_selected_topic(db):

    row_source = db.source['topic_number'].data['Topic'][
        db.source['topic_number'].data['Topic']=='Unnamed # 7'
    ].index
    db.selected_topic(None, None, row_source)


@pytest.mark.parametrize('db', [('selected_sample')], indirect=True)
def test_selected_sample(db):

    row_source = db.source['topic_number'].data['Topic'][
        db.source['topic_number'].data['Topic']=='Unnamed # 7'
    ].index
    db.selected_topic(None, None, row_source)
    db.sample_toggle.active = db.sample_toggle.labels.index('Document Samples')
    db.activate_samples(None)
    db.sample_number.value = 6


@pytest.mark.parametrize('db', [('rename_topic')], indirect=True)
def test_rename_topic(db):

    db.selected_topic(None, None, new=[0])
    db.input['topic_name'].value = 'Topic Renamed'
    db.rename_topic(None)
    db.selected_topic(None, None, new=[1])
    db.selected_topic(None, None, new=[0])


@pytest.mark.parametrize('db', [('get_topic_prediction')], indirect=True)
def test_get_topic_prediction(db):

    db.predict['input'].value = """Baseball season is over. so I'll have more time put my new hard drive in."""
    db.get_topic_prediction(None)


@pytest.mark.parametrize('db', [('recalculate_model')], indirect=True)
def test_recalculate_model(db):

    db.model_inputs['stop_words'].value = 'AX, like'
    db.recalculate_model(None)


@pytest.mark.parametrize('db', [('selected_entity_label')], indirect=True)
def test_selected_entity_label(db):

    row_source = db.source['Entity Label'].data['Terms'][
        db.source['Entity Label'].data['Terms']=='PRODUCT'
    ].index

    db.selected_source(None, None, row_source=row_source, fig_name='Entity Label')

    db.selected_source(None, None, row_source=[0], fig_name='entity')
