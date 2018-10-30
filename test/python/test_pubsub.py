# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_pubsub.py"""
from qiskit import Publisher, Subscriber
from .common import QiskitTestCase


class DummySubscriber(Subscriber):
    """ Simulates a component behaving like a Subscriber """
    def __del__(self):
        self.clear()

    def subscribe(self, event, callback):
        """ Dummy method for activating the subscription to an event.
        Args
            event (string): The event to subsribe
            callback (callable): The callback to execute when the event is emitted """
        super().subscribe(event, callback)


class TestPubSub(QiskitTestCase):
    """A class for testing Publisher/Subscriber functionality.
    """

    def test_pusbsub(self):
        """ Test subscribing works"""
        sub = DummySubscriber()

        def action_callback(test):
            """ Callback called when 'publisher.action` event occurs """
            test.assertTrue(True)

        sub.subscribe("publisher.action", action_callback)
        Publisher().publish("publisher.action", self)

    def test_single_broker(self):
        """ Testing a single broker is instantiated no matter how many
        Publishers or Subscribers we have """

        publishers = [Publisher() for _ in range(10)]
        subscribers = [DummySubscriber() for _ in range(10)]

        for pub, sub in zip(publishers, subscribers):
            self.assertEqual(id(pub._broker), id(sub._broker))

    def test_double_subscribe(self):
        """ Testing that we cannot subscribe the same callback to the same event """
        def callback():
            """ This should be ever called """
            pass

        sub = DummySubscriber()
        sub2 = DummySubscriber()

        sub.subscribe("event", callback)
        self.assertFalse(sub.subscribe("event", callback))
        self.assertFalse(sub2.subscribe("event", callback))

    def test_unsubscribe_simple(self):
        """ Testing a simple unsubscribe works """
        sub = DummySubscriber()

        def callback(_who, test):
            """ This should have ever been called """
            test.assertTrue(False)

        sub.subscribe("publisher.action", callback)
        sub.unsubscribe("publisher.action", callback)
        Publisher().publish("publisher.action", self)
        # As we are not async yet, this should be executed after any possible event handler
        self.assertTrue(True)

    def test_unsubscribe_multiple(self):
        """ Testing unsubscribe works with many other subscribed event works """

        sub = DummySubscriber()

        def callback(test):
            """ This should have ever been called """
            test.assertTrue(False)

        def callback2(_test):
            pass

        sub.subscribe("publisher.action", callback)
        sub.subscribe("publisher.action", callback2)
        sub.unsubscribe("publisher.action", callback)
        Publisher().publish("publisher.action", self)
        # As we are not async yet, this should be executed after any possible event handler
        self.assertTrue(True)

