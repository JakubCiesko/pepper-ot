import sys

import qi


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testrun.py <robot-host>")
        sys.exit(1)
    robot_host = sys.argv[1]
    connection_url = "tcp://%s:9559" % robot_host
    app = qi.Application(["PepperGroundedClientTest", "--qi-url=%s" % connection_url])
    app.start()
    service = app.session.service("PepperGroundedClient")
    print(service.getStatus())
