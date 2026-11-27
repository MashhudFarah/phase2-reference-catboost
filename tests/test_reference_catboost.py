"""Non-fitting checks for reference features, date resolution, and backend wiring."""
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.reference_catboost_features import FEATURE_COLUMNS, engineer_features, usable_feature_mask
from src import reference_catboost_predict as ref


def raw_rows(wards=('A',), start='2025-06-02', weeks=20):
    rows=[]
    dates=pd.date_range(start, periods=weeks, freq='7D')
    for wi,ward in enumerate(wards):
        for i,d in enumerate(dates):
            rows.append({'date':d,'city':'City','ward_id':ward,
                'population_density_sqkm':100+wi,'water_logging_risk_index':.5,'sanitation_score':.6,
                'avg_temp_C':20+i,'max_temp_c':25+i,'min_temp_c':15+i,
                'rainfall_mm':float(i+wi*100),'relative_humidity_pct':50+i,
                'dengue_cases':float(i+wi*100)})
    return pd.DataFrame(rows)


def test_reference_features_are_ordered_and_have_supported_columns():
    data=engineer_features(raw_rows())
    assert list(data.loc[:, FEATURE_COLUMNS].columns)==list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 34
    assert {'dengue_cases_lag_1w','dengue_cases_lag_4w','rain_sum_8w',
            'avg_temp_C_mean_prev_4w','relative_humidity_pct_mean_prev_4w',
            'week_sin','week_cos'} <= set(FEATURE_COLUMNS)
    assert not ({'date','ward_id','city','max_temp_c','min_temp_c',
                 'saline_bags_needed','diagnostic_kits_needed','hospital_beds_required',
                 'target_2w','target_date_2w'} & set(FEATURE_COLUMNS))
    assert usable_feature_mask(data).any()


def test_lags_and_rolls_are_ward_grouped():
    data=engineer_features(raw_rows(('A','B'),weeks=12))
    b=data.loc[data.ward_id.eq('B')].iloc[8]
    assert b.dengue_cases_lag_1w == 107
    assert b.rain_sum_2w == 215


def test_features_do_not_use_future_rows():
    base=raw_rows(weeks=15)
    changed=base.copy()
    changed.loc[changed.date > pd.Timestamp('2025-07-28'), 'dengue_cases']=999999
    left=engineer_features(base).set_index('date').loc[pd.Timestamp('2025-07-28'), FEATURE_COLUMNS]
    right=engineer_features(changed).set_index('date').loc[pd.Timestamp('2025-07-28'), FEATURE_COLUMNS]
    pd.testing.assert_series_equal(left,right)


def test_missing_source_columns_fail_clearly():
    with pytest.raises(ValueError, match='missing source columns'):
        engineer_features(pd.DataFrame({'date': []}))


def test_engineering_rejects_non_monday_and_invalid_humidity():
    non_monday = raw_rows(weeks=10)
    non_monday.loc[0, 'date'] = non_monday.loc[0, 'date'] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='Mondays'):
        engineer_features(non_monday)
    invalid_humidity = raw_rows(weeks=10)
    invalid_humidity.loc[0, 'relative_humidity_pct'] = 101
    with pytest.raises(ValueError, match='relative_humidity_pct'):
        engineer_features(invalid_humidity)


def test_daily_date_resolution_is_on_or_before_and_rejects_outside():
    ward=engineer_features(raw_rows(weeks=20))
    origin,row=ref.resolve_feature_row(ward,'2025-08-14')
    assert origin.date().isoformat()=='2025-08-11'
    assert row.date.date().isoformat()=='2025-08-11'
    with pytest.raises(ValueError,match='no usable feature row'):
        ref.resolve_feature_row(ward,'2025-05-01')
    with pytest.raises(ValueError,match='after the latest'):
        ref.resolve_feature_row(ward,'2025-12-31')


def test_date_resolution_is_ward_specific():
    frame=engineer_features(raw_rows(('A','B'),weeks=20))
    frame=frame.loc[~(frame.ward_id.eq('B') & (frame.date < pd.Timestamp('2025-07-28')))]
    a=frame.loc[frame.ward_id.eq('A')]; b=frame.loc[frame.ward_id.eq('B')]
    assert ref.resolve_feature_row(a,'2025-07-30')[0].date().isoformat()=='2025-07-28'
    assert ref.resolve_feature_row(b,'2025-07-30')[0].date().isoformat()=='2025-07-28'
    assert b.date.min()>a.date.min()


class FakeModel:
    tree_count_=50
    feature_count_=len(FEATURE_COLUMNS)
    def predict(self, matrix):
        assert list(matrix.columns)==list(FEATURE_COLUMNS)
        return np.array([4.375])


@pytest.mark.parametrize('horizon',[2,3,4])
def test_all_horizons_return_raw_decimal_and_target_date(monkeypatch,horizon):
    frame=engineer_features(raw_rows(weeks=22))
    meta={'training_cutoff':'2025-09-01','tree_count':50,'feature_count':len(FEATURE_COLUMNS),
          'model_artifact':f'models/{horizon}w.cbm','metrics':{},'test_rows':1}
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    monkeypatch.setattr(ref,'load_model',lambda h:(FakeModel(),meta,Path('/runs/reference_catboost_demo')))
    monkeypatch.setattr(ref,'derive_city_thresholds',lambda _frame,_city:{'low_max':2,'medium_max':4,'high_max':8})
    result=ref.generate_forecast('City','A',horizon,'2025-08-14')
    assert result['requestedAsOf']=='2025-08-14'
    assert result['observationDate']=='2025-08-11'
    assert result['targetDate']==(pd.Timestamp('2025-08-11')+pd.Timedelta(weeks=horizon)).date().isoformat()
    assert result['rawPredictedCases']==4.375
    assert result['predictionBackend']=='reference_catboost'
    assert result['modelHorizonWeeks']==horizon


def test_retrospective_warning_uses_model_training_cutoff(monkeypatch):
    frame=engineer_features(raw_rows(weeks=22))
    meta={'training_cutoff':'2025-09-01','tree_count':50,'feature_count':len(FEATURE_COLUMNS),
          'model_artifact':'models/2w.cbm','metrics':{},'test_rows':1}
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    monkeypatch.setattr(ref,'load_model',lambda h:(FakeModel(),meta,Path('/runs/reference_catboost_demo')))
    monkeypatch.setattr(ref,'derive_city_thresholds',lambda _frame,_city:{'low_max':2,'medium_max':4,'high_max':8})
    response=ref.generate_forecast('City','A',2,'2025-08-14')
    assert response['isRetrospectiveEstimate'] is True
    assert response['predictionWarning']=='Retrospective historical estimate — not a leakage-safe historical backtest.'


def test_ward_bounds_use_first_and_last_usable_rows(monkeypatch):
    frame=engineer_features(raw_rows(('A','B'),weeks=20))
    frame=frame.loc[~(frame.ward_id.eq('B') & frame.date.eq(frame.loc[frame.ward_id.eq('B'),'date'].min()))]
    monkeypatch.setattr(ref,'load_data',lambda:frame)
    bounds=ref.get_supported_date_bounds()
    for ward in ('A','B'):
        dates=frame.loc[frame.ward_id.eq(ward),'date']
        assert bounds['areas']['City'][ward]=={'min':dates.min().date().isoformat(),'max':dates.max().date().isoformat()}


def test_strict_evaluation_stays_on_legacy_saved_predictions(monkeypatch):
    from src import predict as legacy
    monkeypatch.setattr(legacy,'generate_forecast',lambda *args,**kwargs:{'mode':'strict_evaluation_replay'})
    result=ref.generate_forecast('City','A',2,'2025-08-11',strict_evaluation=True)
    assert result['mode']=='strict_evaluation_replay'
    assert result['predictionBackend']=='phase2_strict_saved_evaluation'


def test_existing_52_feature_runs_are_rejected_by_all_inference_backends():
    from src import live_input_predict as live
    from src import predict as phase2
    from src.live_input_predict import LiveModelUnavailableError
    from src.predict import ModelUnavailableError

    assert ref.get_available_horizons() == []
    assert phase2.get_available_horizons() == []
    with pytest.raises(ref.ModelUnavailableError, match='incompatible'):
        ref.load_model(2)
    with pytest.raises(LiveModelUnavailableError, match='incompatible'):
        live.load_live_model(2)
    with pytest.raises((FileNotFoundError, ModelUnavailableError)):
        phase2.load_model_bundle(2)


def test_shared_inference_loader_accepts_only_canonical_fitted_contract(tmp_path, monkeypatch):
    import json
    import joblib
    from src import live_input_predict as live
    from src import predict as phase2
    from src.feature_engineering import CATEGORICAL_FEATURES
    from src.feature_schema import FEATURE_SCHEMA_VERSION
    from src.train import ARTIFACT_SCHEMA_VERSION, EVALUATION_METHOD, TARGET_DEFINITION

    run = tmp_path / 'phase2_catboost_test_run'
    artifact = run / 'models' / 'model_2w.joblib'
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b'test placeholder')
    metadata = {
        'backend': 'phase2_catboost',
        'artifact_schema_version': ARTIFACT_SCHEMA_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'evaluation_method': EVALUATION_METHOD,
        'target_definition': TARGET_DEFINITION,
        'feature_names': list(FEATURE_COLUMNS),
        'categorical_features': list(CATEGORICAL_FEATURES),
        'horizons': {'2': {
            'horizon_weeks': 2, 'model_artifact': 'models/model_2w.joblib',
            'model_family': 'catboost', 'target_definition': TARGET_DEFINITION,
            'training_data_cutoff': '2023-12-25', 'tree_count': 50,
        }},
    }
    (run / 'metadata.json').write_text(json.dumps(metadata))

    model = FakeModel()
    model.feature_names_ = list(FEATURE_COLUMNS)
    model.tree_count_ = 50
    model.feature_count_ = len(FEATURE_COLUMNS)
    bundle = {
        'artifact_schema_version': ARTIFACT_SCHEMA_VERSION,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'model': model, 'model_family': 'catboost', 'horizon_weeks': 2,
        'feature_names': list(FEATURE_COLUMNS), 'categorical_features': list(CATEGORICAL_FEATURES),
        'target_definition': TARGET_DEFINITION, 'training_data_cutoff': '2023-12-25',
    }
    original_loader = phase2.load_model_bundle
    monkeypatch.setattr(phase2, '_run_directory', lambda _horizon: run)
    monkeypatch.setattr(joblib, 'load', lambda _path: bundle)
    phase2.load_model_bundle.cache_clear()
    try:
        loaded = phase2.load_model_bundle(2)
        assert loaded['model'] is model
        assert loaded['feature_names'] == list(FEATURE_COLUMNS)

        # The reference-style endpoint and live simulator use the same bundle,
        # translating only its diagnostic metadata for response compatibility.
        monkeypatch.setattr(phase2, 'load_model_bundle', lambda _horizon: bundle)
        ref.load_model.cache_clear()
        live.load_live_model.cache_clear()
        ref_model, ref_info, ref_run = ref.load_model(2)
        live_model, live_info, live_run = live.load_live_model(2)
        assert ref_model is model and live_model is model
        assert ref_info['training_cutoff'] == live_info['training_cutoff'] == '2023-12-25'
        assert ref_run == live_run == run
    finally:
        original_loader.cache_clear()
        ref.load_model.cache_clear()
        live.load_live_model.cache_clear()


def test_reference_training_entrypoint_delegates_without_fitting(tmp_path, monkeypatch):
    from src import train_reference_catboost as entrypoint

    calls = []
    expected = tmp_path / 'unwritten-compatible-run'

    def delegate(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(entrypoint, 'run_canonical_training', delegate)
    input_path = tmp_path / 'processed.csv'
    output_parent = tmp_path / 'runs'
    result = entrypoint.run_training(input_path, output_parent)
    assert result == expected
    assert len(calls) == 1
    assert calls[0]['input_path'] == input_path
    assert calls[0]['output_dir'].parent == output_parent
    assert calls[0]['horizons'] == (2, 3, 4)
    assert calls[0]['iterations'] == 50


def test_city_risk_cutoffs_use_only_training_period_city_pool():
    from src.risk_engine import derive_city_thresholds
    frame=pd.DataFrame({
        'date':pd.to_datetime(['2016-01-04','2017-01-02','2018-01-01','2019-01-07',
                               '2020-01-06','2021-01-04','2022-01-03','2023-01-02',
                               '2025-01-06']),
        'city':['A']*8+['A'],
        'ward_id':['w1','w2','w1','w2','w1','w2','w1','w2','w1'],
        'dengue_cases':[1,2,3,4,5,6,7,8,10000],
    })
    thresholds=derive_city_thresholds(frame,'A')
    active=frame.loc[frame.date.dt.year.between(2016,2023)&frame.dengue_cases.gt(0),'dengue_cases']
    assert thresholds=={'low_max':float(active.quantile(.5)),
                        'medium_max':float(active.quantile(.75)),
                        'high_max':float(active.quantile(.9))}
    assert thresholds['high_max']<10000


def test_what_if_uses_copied_consistent_rainfall_features_and_all_horizons(monkeypatch):
    from datetime import date
    from src import live_input_predict as live
    from src import risk_engine
    from src.backend import operational_data

    raw=raw_rows(('A',),start='2016-01-04',weeks=32)
    history=engineer_features(raw)
    history=history.loc[usable_feature_mask(history)].copy()
    monkeypatch.setattr(live,'load_historical_source',lambda:history)
    run=Path('/runs/phase2-dengue-34-v1-test')
    inputs=[]
    class RainModel:
        def predict(self,matrix):
            inputs.append(matrix.copy())
            return np.array([float(matrix['rainfall_mm'].iloc[0]+matrix['rain_sum_2w'].iloc[0])])
    models={h:(RainModel(),{'training_cutoff':'2016-06-01','model_artifact':f'{h}w.cbm'},run)
            for h in (2,3,4)}
    monkeypatch.setattr(live,'load_live_model',lambda horizon:models[horizon])
    capacity=pd.DataFrame([{'ward_id':'A','city':'City','vulnerability_index':.6}])
    monkeypatch.setattr(operational_data,'read_ward_capacity',lambda:capacity)
    monkeypatch.setattr(risk_engine,'load_risk_calibration_data',lambda _path:raw[['date','city','dengue_cases']].assign(date=pd.to_datetime(raw.date)))
    result=live.simulate_what_if(city='City',ward_ids=['A'],forecast_date=date(2016,8,7),
                                 hypothetical_crew_count=2,intervention_effectiveness_pct=25)
    assert len(result['forecasts'])==3
    assert {row['horizon_weeks'] for row in result['forecasts']}=={2,3,4}
    assert all(row['target_date']==(pd.Timestamp(row['forecast_origin'])+pd.Timedelta(weeks=row['horizon_weeks'])).date().isoformat()
               for row in result['forecasts'])
    assert all(row['scenario_expected_cases']<row['baseline_expected_cases'] for row in result['forecasts'])
    assert all(row['hypothetical_crews_assigned']==2 for row in result['forecasts'])
    assert len(inputs)==6
    before,after=inputs[:2]
    factor=.75
    for name in ('rainfall_mm','rain_lag_1w','rain_lag_2w','rain_lag_3w','rain_lag_4w',
                 'rain_sum_2w','rain_sum_3w','rain_sum_4w','rain_sum_6w','rain_sum_8w'):
        assert after[name].iloc[0]==pytest.approx(before[name].iloc[0]*factor)
    assert result['rainfall_proxy_factor']==factor
    assert 'not causal' in result['warning']


def test_frontend_date_input_allows_daily_dates_and_horizon_keeps_selection():
    root=Path(__file__).resolve().parents[1]
    html=(root/'templates/user/user_dashboard.html').read_text()
    js=(root/'static/js/user/user.js').read_text()
    assert 'type="date"' in html
    assert not any('step=' in line for line in html.splitlines() if 'type="date"' in line)
    assert '/static/js/user/user.js?v=11' in html
    assert 'A date between weekly observations resolves to the latest observation on or before your selected date.' in html
    assert 'asOfDate.min = bounds.min;' in js and 'asOfDate.max = bounds.max;' in js
    city_start=js.index('citySelect.addEventListener("change"')
    ward_start=js.index('wardSelect.addEventListener("change"')
    horizon_start=js.index('forecastForm.querySelectorAll', ward_start)
    current_start=js.index('currentMode.addEventListener', horizon_start)
    assert 'updateDateBounds();' in js[city_start:ward_start]
    assert 'updateDateBounds();' in js[ward_start:horizon_start]
    horizon_listener=js[horizon_start:current_start]
    assert 'updateDateBounds' not in horizon_listener and 'asOfDate.value' not in horizon_listener
    assert 'timeZone: "UTC"' in js
    assert 'const FORECAST_TIMEOUT_MS = 30000;' in js
    assert 'invalidatePendingForecast();' in js
    assert 'controller.abort()' in js and 'window.clearTimeout(timeout)' in js
    assert 'citySelect.disabled = isLoading' not in js
