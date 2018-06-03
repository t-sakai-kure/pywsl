import time

class Timer:
  def __init__(self):
    self.record = False
    self.ptime  = float('nan')

  def tic(self, message=None):
    if message is not None:
      print(message)
    self.record = True
    self.ptime  = time.time()

  def toc(self, quiet=False):
    if not self.record:
      print('tic is not called')
      return self.ptime

    etime = time.time() - self.ptime
    self.record = False
    self.ptime  = float('nan')
    if not quiet:
      print('Elapsed time: {:.2f} [sec]'.format(etime))
    return etime

