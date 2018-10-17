# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_pubsub.py"""
from qiskit import Publisher, Subscriber
from .common import QiskitTestCase


class TestSubscriber(Subscriber):
    """ Simulates a component behaving like a Subscriber """
    def subscribe_event(self, event, callback):
        """ Dummy method for activating the subscription to an event.
        Args
            event (string): The event to subsribe
            callback (callable): The callback to execute when the event is emitted """
        self.subscribe(event, callback)


class TestPublisher(Publisher):
    """ Simulates a component behaving like a Publisher """
    def action(self, test):
        """ Dummy method to trigger an event.
         Args
            test: Dummy parameter to pass to the emitted"""
        self.publish("publisher.action", self.__class__, test)


class TestPubSub(QiskitTestCase):
    """A class for testing Publisher/Subscriber functionality.
    """

    def test_pusbsub(self):
        """ Test subscribing works"""
        pub = TestPublisher()
        sub = TestSubscriber()

        def action_callback(who, test):
            """ Callback called when 'publisher.action` event occurs """
            test.assertTrue(who == TestPublisher)

        sub.subscribe_event("publisher.action", action_callback)
        pub.action(self)

    def test_single_broker(self):
        """ Testing a single broker is instantiated no matter how many
        Publishers or Subscribers we have """

        publishers = [TestPublisher() for _ in range(10)]
        subscribers = [TestSubscriber() for _ in range(10)]

        for pub, sub in zip(publishers, subscribers):
            self.assertEqual(id(pub._broker), id(sub._broker))

    def test_double_subscribe(self):
        """ Testing that we cannot subscribe the same callback to the same event """
        def callback():
            """ This should be ever called """
            pass

        sub = TestSubscriber()
        sub2 = TestSubscriber()

        sub.subscribe("event", callback)
        self.assertFalse(sub.subscribe("event", callback))
        self.assertFalse(sub2.subscribe("event", callback))
