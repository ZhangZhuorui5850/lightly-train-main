SELECT *
FROM (
    VALUES
        (1, '速度校准', 3000, 3.7, 4, 6, '确认显存、step time、loss 和 mAP 能正常变化'),
        (2, '本机基线', 20000, 24.8, 26, 38, '形成可信的 640 基线'),
        (3, '延长基线', 30000, 37.3, 39, 56, '仅在 20k 时 AP_small 仍持续上升时续训'),
        (4, '200 有效 epoch', 161200, 200.0, 216, 312, '本机约 9–13 天，优先留给服务器')
) AS plan(run_order, experiment, steps, effective_epochs, hours_low, hours_high, decision_rule);
