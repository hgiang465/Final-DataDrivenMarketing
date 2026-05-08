import pytest


def segment_churn_risk(probability):
    """
    Assign churn risk segment based on probability
    """
    if probability >= 0.7:
        return 'High Risk'
    elif probability >= 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'


class TestSegmentationLogic:
    
    def test_high_risk_segment(self):
        """Test high risk threshold"""
        assert segment_churn_risk(0.7) == 'High Risk'
        assert segment_churn_risk(0.85) == 'High Risk'
        assert segment_churn_risk(1.0) == 'High Risk'
    
    def test_medium_risk_segment(self):
        """Test medium risk threshold"""
        assert segment_churn_risk(0.4) == 'Medium Risk'
        assert segment_churn_risk(0.5) == 'Medium Risk'
        assert segment_churn_risk(0.69) == 'Medium Risk'
    
    def test_low_risk_segment(self):
        """Test low risk threshold"""
        assert segment_churn_risk(0.0) == 'Low Risk'
        assert segment_churn_risk(0.2) == 'Low Risk'
        assert segment_churn_risk(0.39) == 'Low Risk'
    
    def test_boundary_cases(self):
        """Test exact boundary values"""
        assert segment_churn_risk(0.4) == 'Medium Risk'
        assert segment_churn_risk(0.7) == 'High Risk'
        assert segment_churn_risk(0.3999) == 'Low Risk'
        assert segment_churn_risk(0.6999) == 'Medium Risk'