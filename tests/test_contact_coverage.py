from conveyor_bench.isaac.finger_contact_tensor_probe import support_coverage_result


def test_table_contact_does_not_prove_articulation_contact_coverage():
    result=support_coverage_result(0,[True,True],[False,False],external_witness=True,articulated_witness=False)
    assert result==(None,True,False,False)


def test_direct_support_witness_is_positive_but_missing_data_cannot_prove_absence():
    assert support_coverage_result(2,[False,False],[False,False],external_witness=False,articulated_witness=False)[0] is True
    assert support_coverage_result(0,[False,False],[False,False],external_witness=False,articulated_witness=False)[0] is None


def test_validated_reciprocal_contacts_allow_absence_until_a_mismatch():
    assert support_coverage_result(0,[True,True],[True,True],external_witness=True,articulated_witness=False)[0] is False
    assert support_coverage_result(0,[True,True],[False,False],external_witness=True,articulated_witness=True)[0] is None
