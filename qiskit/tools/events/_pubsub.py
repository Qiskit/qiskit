# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""
Message broker for the Publisher / Subscriber mechanism
"""

from qiskit.qiskiterror import QiskitError


class _Broker(object):
    """The event/message broker. It's a singleton.

    In order to keep consistency across all the components, it would be great to
    have a specific format for new events, documenting their usage.
    It's the responsibility of the component emitting an event to document it's usage in
    the component docstring.

    Event format:
        "terra.<component>.<method>.<action>"

    Examples:
        "terra.transpiler.compile.start"
        "terra.job.status.changed"
        "terra.backend.run.start"
        "terra.job.result.received"
    """

    _instance = None
    _subscribers = {}

    def __new__(cls):
        if _Broker._instance is None:
            _Broker._instance = object.__new__(cls)
        return _Broker._instance

    class _Subscription:
        def __init__(self, event, callback):
            self.event = event
            self.callback = callback

        def __eq__(self, other):
            """Overrides the default implementation"""
            if isinstance(other, self.__class__):
                return self.event == other.event and \
                       self.callback.__name__ == other.callback.__name__
            return False

    def subscribe(self, event, callback):
        """Subscribes to an event, so when it's emitted all the callbacks subscribed,
        will be executed. We are not allowing double registration.

        Args
            event (string): The event to subscribed in the form of:
                            "terra.<component>.<method>.<action>"
            callback (callable): The callback that will be executed when an event is
                                  emitted.
        """
        if not callable(callback):
            raise QiskitError("Callback is not a callable!")

        if event not in self._subscribers:
            self._subscribers[event] = []

        new_subscription = self._Subscription(event, callback)
        if new_subscription in self._subscribers[event]:
            # We are not allowing double subscription
            return False

        self._subscribers[event].append(new_subscription)
        return True

    def dispatch(self, event, *args, **kwargs):
        """Emits an event if there are any subscribers.

        Args
            event (String): The event to be emitted
            args: Arguments linked with the event
            kwargs: Named arguments linked with the event
        """
        # No event, no subscribers.
        if event not in self._subscribers:
            return

        for subscriber in self._subscribers[event]:
            subscriber.callback(*args, **kwargs)

    def unsubscribe(self, event, callback):
        """ Unsubscribe the specific callback to the event.

        Args
            event (String): The event to unsubscribe
            callback (callable): The callback that won't be executed anymore

        Returns
            True: if we have successfully unsubscribed to the event
            False: if there's no callback previously registered
        """

        try:
            self._subscribers[event].remove(self._Subscription(event, callback))
        except KeyError:
            return False

        return True

    def clear(self):
        """ Unsubscribe everything, leaving the Broker without subscribers/events.
        """
        self._subscribers.clear()


class Publisher(object):
    """ Represents a Publisher, every component (class) can become a Publisher and
    send events by inheriting this class. Functions can call this class like:
    Publisher().publish("event", args, ... )
    """
    def __init__(self):
        self._broker = _Broker()

    def publish(self, event, *args, **kwargs):
        """ Triggers an event, and associates some data to it, so if there are any
        subscribers, their callback will be called synchronously. """
        return self._broker.dispatch(event, *args, **kwargs)


class Subscriber(object):
    """ Represents a Subscriber, every component (class) can become a Subscriber and
    subscribe to events, that will call callback functions when they are emitted.
    """
    def __init__(self):
        self._broker = _Broker()

    def subscribe(self, event, callback):
        """ Subscribes to an event, associating a callback function to that event, so
        when the event occurs, the callback will be called.
        This is a blocking call, so try to keep callbacks as lighweight as possible. """
        return self._broker.subscribe(event, callback)

    def unsubscribe(self, event, callback):
        """ Unsubscribe a pair event-callback, so the callback will not be called anymore
        when the event occurs."""
        return self._broker.unsubscribe(event, callback)

    def clear(self):
        """ Unsubscribe everything"""
        self._broker.clear()
