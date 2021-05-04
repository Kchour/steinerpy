import unittest
import io
import sys
import steinerpy.config as cfg

# # TESTING PURPOSE ONLY: store console streams, assign new stream capturing objects
# old_out, old_err = sys.stdout, sys.stderr
# new_out, new_err = io.StringIO(), io.StringIO()

@unittest.skip("SKIPPING DUE TO WEIRD INTERACTION WITH UNITTEST MODULE")
class TestLogger(unittest.TestCase):
    """Test helper class MyLogger """

    # def setUp(self):
    #     old_out, old_err = sys.stdout, sys.stderr
    #     new_out, new_err = io.StringIO(), io.StringIO()

    def test_create_logger(self):
        # TESTING PURPOSE ONLY: redirect out,err to capturing objects
        old_out, old_err = sys.stdout, sys.stderr
        new_out, new_err = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = new_out, new_err 

        # Configure console level to following choices before importing Logger:
        cfg.Misc.console_level = "DEBUG"
        from steinerpy.library.logger import MyLogger

        loggerName = "test_logger"

        msgDebug = "This is a debug level"
        input_level = "debug"
        MyLogger.add_message(msgDebug, loggerName, input_level)

        msgInfo = "This is info level"
        input_level = "info"
        MyLogger.add_message(msgInfo, loggerName, input_level)

        msgWarn = "This is warning level"
        input_level = "warning"
        MyLogger.add_message(msgWarn, loggerName, input_level)
        
        msgErr = "This is error level"
        input_level = "error"
        MyLogger.add_message(msgErr, loggerName, input_level)

        msgCrit = "This is critical level"
        input_level = "critical"
        MyLogger.add_message(msgCrit, loggerName, input_level)    

        # TESTING PURPOSE ONLY: redirect out, err back to terminal stream
        sys.stdout, sys.stderr = old_out, old_err 

        # # print captured data
        # print(new_err.getvalue().strip(), new_out.getvalue())

        self.assertTrue("- test_logger - DEBUG - \x1b[92mThis is a debug level\x1b[0m" in new_err.getvalue())
        self.assertTrue("- test_logger - INFO - \x1b[94mThis is info level\x1b[0m" in new_err.getvalue())
        self.assertTrue("- test_logger - WARNING - \x1b[93mThis is warning level\x1b[0m" in new_err.getvalue())
        self.assertTrue("- test_logger - ERROR - \x1b[91mThis is error level\x1b[0m" in new_err.getvalue())
        self.assertTrue("- test_logger - CRITICAL - \x1b[31mThis is critical level\x1b[0m" in new_err.getvalue())

if __name__ == "__main__":
    unittest.main()
