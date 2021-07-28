import winreg


def _get_guid() -> str:
    """
    get reg guid
    :return:
    """
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\Microsoft\\Cryptography", 0,
                         winreg.KEY_QUERY_VALUE | winreg.KEY_WOW64_64KEY)
    return winreg.QueryValueEx(key, "MachineGuid")[0]


GUID = _get_guid()