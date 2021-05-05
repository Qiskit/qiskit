# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for qiskit/tools/events/_pubsub.py"""

from qiskit.tools.events.pubsub import Publisher, Subscriber
from qiskit.test import QiskitTestCase


class DummySubscriber(Subscriber):
    """Simulates a component behaving like a Subscriber"""

    def __del__(self):
        self.clear()


class TestPubSub(QiskitTestCase):
    """A class for testing Publisher/Subscriber functionality."""

    def test_pusbsub(self):
        """Test subscribing works"""
        sub = DummySubscriber()

        def action_callback(test):
            """Callback called when 'publisher.action` event occurs"""
            test.assertTrue(True)

        sub.subscribe("publisher.action", action_callback)
        Publisher().publish("publisher.action", self)

    def test_single_broker(self):
        """Testing a single broker is instantiated no matter how many
        Publishers or Subscribers we have"""

        publishers = [Publisher() for _ in range(10)]
        subscribers = [DummySubscriber() for _ in range(10)]

        for pub, sub in zip(publishers, subscribers):
            self.assertEqual(id(pub._broker), id(sub._broker))

    def test_double_subscribe(self):
        """Testing that we cannot subscribe the same callback to the same event"""

        def callback():
            """This should be ever called"""
            pass

        sub = DummySubscriber()
        sub2 = DummySubscriber()

        sub.subscribe("event", callback)
        self.assertFalse(sub.subscribe("event", callback))
        self.assertFalse(sub2.subscribe("event", callback))

    def test_unsubscribe_simple(self):
        """Testing a simple unsubscribe works"""
        sub = DummySubscriber()

        def callback(_who, test):
            """This should have ever been called"""
            test.fail("We shouldn't have reach this code!")

        sub.subscribe("publisher.action", callback)
        sub.unsubscribe("publisher.action", callback)
        Publisher().publish("publisher.action", self)

    def test_unsubscribe_multiple(self):
        """Testing unsubscribe works with many other subscribed event works"""

        sub = DummySubscriber()

        def callback(test):
            """This should have ever been called"""
            test.fail("We shouldn't have reach this code!")

        def dummy_callback(_test):
            """Just a dummy callback, it won't be executed"""
            pass

        sub.subscribe("publisher.action", callback)
        sub.subscribe("publisher.action", dummy_callback)
        sub.unsubscribe("publisher.action", callback)
        Publisher().publish("publisher.action", self)
