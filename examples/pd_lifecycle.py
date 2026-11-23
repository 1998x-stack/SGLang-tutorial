"""Toy PD sender/receiver lifecycle; does not perform network or GPU transfers."""
from dataclasses import dataclass
class TransitionError(ValueError): pass
@dataclass
class Request:
    policy: str = "early"
    handshaken: bool = False
    prefill_done: bool = False
    destination_allocated: bool = False
    transferred: bool = False
    aborted: bool = False
    def handshake(self):
        if self.aborted: raise TransitionError("aborted request")
        self.handshaken=True
        if self.policy=="early": self.destination_allocated=True
    def finish_prefill(self):
        if not self.handshaken or self.aborted: raise TransitionError("handshake required")
        self.prefill_done=True
    def allocate_destination(self):
        if not self.handshaken or self.aborted: raise TransitionError("handshake required")
        if self.policy=="prefill_complete" and not self.prefill_done: raise TransitionError("wait for prefill completion")
        self.destination_allocated=True
    def transfer(self):
        if self.aborted or not self.handshaken or not self.prefill_done or not self.destination_allocated:
            raise TransitionError("transfer preconditions not met")
        self.transferred=True
    def abort(self):
        if self.transferred: raise TransitionError("successful transfer already committed")
        self.aborted=True
if __name__=="__main__":
    r=Request("prefill_complete"); r.handshake(); r.finish_prefill(); r.allocate_destination(); r.transfer()
    print("prefill_complete lifecycle:",r)
