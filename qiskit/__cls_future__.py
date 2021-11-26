all_feature_names = ["mcgates"]

__all__ = ["all_feature_names"] + all_feature_names

MC_GATES = 0x0010


class _QiskitFuture:
    def __init__(self, optional_release, mandatory_release, flag):
        self.optional_release = optional_release
        self.mandatory_release = mandatory_release
        self.flag = flag


mcgates = _QiskitFuture((0, 19, 0), (0, 20, 0), MC_GATES)
