from soda_stt import SodaRecognizer, _config
_config.LANGUAGE_PACK_IDS.update({'vi-VN':'bpnckojighgibleldbofdjmfdipfeiak','ja-JP':'onhpjgkfgajmkkeniaoflicgokpaebfa'})
for loc in ('vi-VN','ja-JP'):
    try:
        print('INIT',loc,flush=True)
        rec=SodaRecognizer.for_locale(loc)
        print('READY',loc,flush=True)
    except Exception as exc: print('FAILED',loc,type(exc).__name__,str(exc)[:500],flush=True)
