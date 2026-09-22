"""Recording indicator: a small pill above all windows with a pulsing dot and a timer.

AppKit windows need the main thread of their process and a run loop on it, and
the main thread of `s2t serve` belongs to uvicorn. So the pill lives in a tiny
helper process (`python -m s2t.overlay ...`) that is spawned once and driven
over stdin: "show" / "hide" lines, EOF = exit (so it can never outlive us).

The helper must never take keyboard focus - the dictated text is pasted into
whatever window the user was in. Hence: accessory app (no Dock icon, never
activated), non-activating borderless panel, mouse events ignored.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time

from .config import OverlayConfig

POSITIONS = ("top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right")


def format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}:{secs:02d}"


def pill_origin(
    position: str, area: tuple[float, float, float, float], size: tuple[float, float], margin: float
) -> tuple[float, float]:
    """Bottom-left corner of the pill inside `area` (x, y, width, height). AppKit's y axis points up."""
    x, y, width, height = area
    if position.endswith("left"):
        left = x + margin
    elif position.endswith("right"):
        left = x + width - size[0] - margin
    else:
        left = x + (width - size[0]) / 2
    bottom = y + height - size[1] - margin if position.startswith("top") else y + margin
    return left, bottom


class Overlay:
    """Parent-side handle. Every method is a no-op when the overlay is disabled."""

    def __init__(self, config: OverlayConfig) -> None:
        if config.position not in POSITIONS:
            raise ValueError(
                f"Unknown dictation.overlay.position: {config.position!r} (use one of {', '.join(POSITIONS)})"
            )
        self.config = config
        self._process: subprocess.Popen | None = None

    def start(self) -> None:
        """Spawn the helper ahead of the first key press (importing AppKit takes ~0.2 s)."""
        if not self.config.enabled or (self._process is not None and self._process.poll() is None):
            return
        self._process = subprocess.Popen(
            [sys.executable, "-m", "s2t.overlay",
             self.config.position, str(self.config.margin), str(self.config.scale)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        )

    def show(self) -> None:
        self._send("show")

    def hide(self) -> None:
        self._send("hide")

    def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        try:
            process.stdin.close()  # EOF makes the helper exit
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()

    def _send(self, command: str) -> None:
        # Called from the hotkey listener: must not raise and must not block.
        if not self.config.enabled:
            return
        try:
            self.start()  # respawns a helper that died
            self._process.stdin.write(f"{command}\n".encode())
            self._process.stdin.flush()
        except OSError as error:
            print(f"[overlay] {error}", flush=True)
            self._process = None


# -- helper process ----------------------------------------------------------


def _serve(position: str, margin: float, scale: float) -> None:
    import AppKit
    from PyObjCTools import AppHelper
    from Quartz import CABasicAnimation

    app = AppKit.NSApplication.sharedApplication()
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    height, dot_size, pad, gap = 30 * scale, 10 * scale, 12 * scale, 8 * scale

    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        AppKit.NSMakeRect(0, 0, 100, height),
        AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel,
        AppKit.NSBackingStoreBuffered,
        False,
    )
    panel.setLevel_(AppKit.NSScreenSaverWindowLevel)  # above full-screen apps and the menu bar
    panel.setCollectionBehavior_(
        AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | AppKit.NSWindowCollectionBehaviorStationary
        | AppKit.NSWindowCollectionBehaviorIgnoresCycle
    )
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setHasShadow_(False)
    panel.setIgnoresMouseEvents_(True)
    panel.setHidesOnDeactivate_(False)

    pill = panel.contentView()
    pill.setWantsLayer_(True)
    pill.layer().setBackgroundColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(0.08, 0.85).CGColor())
    pill.layer().setCornerRadius_(height / 2)
    pill.layer().setBorderWidth_(1)
    pill.layer().setBorderColor_(AppKit.NSColor.colorWithCalibratedWhite_alpha_(1.0, 0.18).CGColor())

    dot = AppKit.NSView.alloc().initWithFrame_(
        AppKit.NSMakeRect(pad, (height - dot_size) / 2, dot_size, dot_size)
    )
    dot.setWantsLayer_(True)
    dot.layer().setBackgroundColor_(AppKit.NSColor.systemRedColor().CGColor())
    dot.layer().setCornerRadius_(dot_size / 2)
    pill.addSubview_(dot)

    label = AppKit.NSTextField.labelWithString_("")
    label.setFont_(AppKit.NSFont.monospacedDigitSystemFontOfSize_weight_(14 * scale, AppKit.NSFontWeightMedium))
    label.setTextColor_(AppKit.NSColor.whiteColor())
    pill.addSubview_(label)

    state = {"started": 0.0, "timer": None, "template": ""}

    def target_area() -> tuple[float, float, float, float]:
        # The screen under the mouse pointer, minus the menu bar and the Dock.
        mouse = AppKit.NSEvent.mouseLocation()
        screen = next(
            (s for s in AppKit.NSScreen.screens() if AppKit.NSMouseInRect(mouse, s.frame(), False)),
            AppKit.NSScreen.mainScreen(),
        )
        frame = screen.visibleFrame()
        return frame.origin.x, frame.origin.y, frame.size.width, frame.size.height

    def render() -> None:
        text = format_elapsed(time.monotonic() - state["started"])
        template = re.sub(r"\d", "0", text)
        if template != state["template"]:  # first frame, or the text got longer: 9:59 -> 10:00
            state["template"] = template
            label.setStringValue_(template)
            label.sizeToFit()
            size = label.frame().size
            label.setFrameOrigin_((pad + dot_size + gap, (height - size.height) / 2))
            width = pad + dot_size + gap + size.width + pad
            left, bottom = pill_origin(position, target_area(), (width, height), margin)
            panel.setFrame_display_(AppKit.NSMakeRect(left, bottom, width, height), True)
        label.setStringValue_(text)

    def show() -> None:
        hide()
        state.update(started=time.monotonic(), template="")
        render()
        pulse = CABasicAnimation.animationWithKeyPath_("opacity")
        pulse.setFromValue_(1.0)
        pulse.setToValue_(0.25)
        pulse.setDuration_(0.7)
        pulse.setAutoreverses_(True)
        pulse.setRepeatCount_(float("inf"))
        dot.layer().addAnimation_forKey_(pulse, "pulse")
        panel.orderFrontRegardless()  # shows the window without activating the app
        state["timer"] = AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            0.1, True, lambda _timer: render()
        )

    def hide() -> None:
        if state["timer"] is not None:
            state["timer"].invalidate()
            state["timer"] = None
        dot.layer().removeAnimationForKey_("pulse")
        panel.orderOut_(None)

    actions = {"show": show, "hide": hide}

    def read_commands() -> None:
        for line in sys.stdin:
            action = actions.get(line.strip())
            if action is not None:
                AppHelper.callAfter(action)
        os._exit(0)  # stdin closed: the parent is gone

    threading.Thread(target=read_commands, daemon=True).start()
    app.run()


def main() -> None:
    # Ctrl+C in the terminal reaches the whole process group; the parent shuts us down via EOF.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    position, margin, scale = sys.argv[1:4]
    _serve(position, float(margin), float(scale))


if __name__ == "__main__":
    main()
