"""Historical contour thresholds used throughout the clinical review screens."""
RISK_SOURCE = '既有边缘不规则指数阈值，未经临床校准；七分类候选独立展示'


def rule_level(irregularity):
    if irregularity is None:
        return '未知'
    if irregularity > 1.5:
        return '高风险'
    if irregularity > 1.3:
        return '中-高风险'
    if irregularity > 1.0:
        return '中风险'
    return '低风险'
