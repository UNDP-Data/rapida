import logging

logger = logging.getLogger(__name__)

class CustomStreamHandler(logging.StreamHandler):
  """Handler that controls the writing of the newline character"""

  special_code = '[!n]'
  active = False
  def emit(self, record) -> None:

    if self.special_code in record.msg:
      record.msg = record.msg.replace( self.special_code, '' )
      self.terminator = ''
      if not self.active:
        self.active = True
      self.stream.write('\r')
      self.flush()
    else:
        self.terminator = '\n'
        if self.active:
            self.stream.write(self.terminator)
            self.flush()
        self.active = False
    try:
        msg = self.format(record)
        # issue 35046: merged two stream.writes into one.
        self.stream.write(msg + self.terminator)
        self.flush()
    except RecursionError:  # See issue 36272
        raise
    except Exception:
        self.handleError(record)
