"""CPU UTF-8 and continuity toy; not a tokenizer or SGLang DetokenizeState."""
import codecs
def decode_chunks(chunks):
    decoder=codecs.getincrementaldecoder("utf-8")("strict")
    out=[decoder.decode(c,final=False) for c in chunks]
    out.append(decoder.decode(b"",final=True))
    return out
def continuous(received,new,total):
    return received>=0 and new>=0 and received+new==total
def selftest():
    assert decode_chunks([b"\xe4",b"\xb8",b"\xad",b""])==["","","中","",""]
    data="A😊B".encode()
    assert "".join(decode_chunks([bytes([b]) for b in data]))=="A😊B"
    assert continuous(2,1,3) and not continuous(2,1,4)
    try: decode_chunks([b"\xe4"])
    except UnicodeDecodeError: pass
    else: raise AssertionError("incomplete stream")
if __name__=="__main__":
    selftest()
    print(decode_chunks([b"\xe4",b"\xb8",b"\xad"]))
