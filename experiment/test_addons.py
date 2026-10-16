import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from addons import motion, LowerTail, stall_score, predict, measure

def test_startup_has_no_future_fill():
    g = np.eye(10, dtype=np.float32)
    assert np.isnan(motion(g, 4)[:4]).all()
    theta = np.arange(10) / 10
    tails = {lag: LowerTail().fit(motion(g, lag), theta) for lag in [1,4]}
    full = stall_score(g, theta, tails)
    assert not full[:4].any()
    for n in range(1,11):
        np.testing.assert_array_equal(stall_score(g[:n], theta[:n], tails), full[:n])

def test_six_arms_are_separate_and_causal():
    base = np.full(40, 2.)
    stall = np.r_[np.zeros(5), np.full(35, 2.)]
    fitted = dict(base_threshold=3., stall_threshold=1., fusion_threshold=3., base_low=0., fusion_low=0.)
    s, a, t = predict(base, stall, fitted)
    assert not a[:,0].any() and not a[:,1].any()
    assert not a[:7,2].any() and a[7:,2].all()
    assert not a[:,3].any() and a[7:,5].all()
    np.testing.assert_array_equal(s[:,:3], np.repeat(base[:,None],3,axis=1))
    assert t[22,1] == 3. and t[23,1] == 3.6
    for n in range(1,41):
        for prefix, full in zip(predict(base[:n],stall[:n],fitted),(s,a,t)):
            np.testing.assert_array_equal(prefix,full[:n])

def test_unknown_frames_do_not_enter_f1_or_fpr():
    m = measure(np.array([0,1,1,-1]), np.array([.1,.9,.8,100]), np.array([False,True,False,True]))
    assert m['tp']==1 and m['fp']==0 and m['fn']==1 and m['tn']==1
    assert m['f1']==2/3 and m['fpr']==0 and m['recall']==.5
