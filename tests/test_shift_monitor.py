from phystta.monitoring import ShiftMonitor, detection_delay


def test_shift_monitor_alerts_after_consecutive_exceedances():
    mon = ShiftMonitor(alert_quantile=0.9, min_consecutive_alerts=2)
    mon.calibrate({"mmd": [0.1, 0.12, 0.09, 0.11, 0.1, 0.13]})
    e1 = mon.update("mmd", 0.2, 0)
    e2 = mon.update("mmd", 0.21, 1)
    assert e1.alert is False
    assert e2.alert is True


def test_detection_delay():
    mon = ShiftMonitor(alert_quantile=0.9, min_consecutive_alerts=1)
    mon.calibrate({"mmd": [0.1, 0.1, 0.1, 0.1, 0.1]})
    events = [mon.update("mmd", s, i) for i, s in enumerate([0.05, 0.07, 0.2])]
    d = detection_delay(events, drift_start_step=1)
    assert d >= 0
