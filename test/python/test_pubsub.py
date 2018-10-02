# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for qiskit/_pubsub.py"""
from .common import QiskitTestCase

from qiskit import Publisher, Subscriber


class TestSubscriber(Subscriber):
    def subscribe_event(self, event, callback):
        self.subscribe(event, callback)


class TestPublisher(Publisher):
    def action(self, test):
        self.publish("publisher.action", self.__class__, test)


class TestPubSub(QiskitTestCase):
    """A class for testing Publisher/Subscriber functionality.
    """

    def test_pusbsub(self):
        """ Test subscribing works"""
        pub = TestPublisher()
        sub = TestSubscriber()

        def action_callback(who, test):
            test.assertTrue(who == TestPublisher)

        sub.subscribe_event("publisher.action", action_callback)
        pub.action(self)

    def test_single_broker(self):
        """ Testing a single broker is instantiated no matter how many Publishers or Subsribers we have """

        publishers = [TestPublisher() for _ in range(10)]
        subscribers= [TestSubscriber() for _ in range(10)]

        for pub,sub in zip(publishers, subscribers):
            self.assertEqual(id(pub._broker), id(sub._broker))

    def test_double_subscribe(self):
        """ Testing that we cannot subscribe the same callback to the same event """
        def callback():
            pass

        sub = TestSubscriber()
        sub2 = TestSubscriber()

        sub.subscribe("event", callback)
        self.assertFalse(sub.subscribe("event", callback))
        self.assertFalse(sub2.subscribe("event", callback))
