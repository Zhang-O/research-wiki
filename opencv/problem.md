- 远程服务器报错，无法正确显示图像
```
    qt.qpa.xcb: X server does not support XInput 2
    failed to get the current screen resources
    qt.qpa.xcb: QXcbConnection: XCB error: 1 (BadRequest), sequence: 165, resource id: 90, major code: 130 (Unknown), minor code: 47
    qt.qpa.xcb: QXcbConnection: XCB error: 170 (Unknown), sequence: 178, resource id: 90, major code: 146 (Unknown), minor code: 20
    The X11 connection broke (error 1). Did the X11 server die?
```
    可能是由于 python 环境对应的 opencv (4.4.0) 版本的问题， 使用另一个 python 环境 opencv (4.1.0) 就不会出现问题。
