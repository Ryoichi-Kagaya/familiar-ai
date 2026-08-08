# TapoカメラをTailscale越しに使う

複数カメラは `~/.familiar_ai/cameras.json` にまとめて登録できます。通常の `see()` /
`look()` は既定カメラを使い、`see_camera(camera=...)` / `look_camera(camera=...)` で同じ
起動中に別のカメラへ切り替えられます。外出用カメラはオンデマンド接続にできるため、
使っていない間に接続エラーを繰り返しません。

## 構成

Tapoカメラ自体にはTailscaleをインストールできません。そのため、2台目のカメラと
同じLANに常時起動するLinux端末（Raspberry Pi、ミニPCなど）を1台置き、Tailscaleの
subnet routerにします。

```text
familiar-ai PC + Tailscale
            |
       encrypted tailnet
            |
camera側Linux端末（subnet router） ── 2台目Tapo（LAN IP）
```

LAN全体ではなくカメラ1台だけを公開するため、カメラのIPを `/32` でadvertiseします。

## 1. このPCをTailnetへ接続する

このUbuntu PCで実行します。インストール時にsudoパスワードが必要です。

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
sudo tailscale set --accept-routes
```

表示されたURLをブラウザで開き、Tailnetへログインします。

## 2. カメラ側にsubnet routerを用意する

以下は、2台目のカメラと同じLANに置いたLinux端末で実行します。
`<CAMERA_LAN_IP>` はカメラの固定LAN IPへ置き換えます。

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up

echo 'net.ipv4.ip_forward = 1' | sudo tee /etc/sysctl.d/99-tailscale.conf
sudo sysctl -p /etc/sysctl.d/99-tailscale.conf

sudo tailscale set --advertise-routes=<CAMERA_LAN_IP>/32
```

次にTailscaleの管理画面で、カメラ側端末の **Subnets → Edit route settings** を開き、
advertiseされた `<CAMERA_LAN_IP>/32` を承認します。

> カメラ側ルーターのDHCP予約などを使い、2台目TapoのLAN IPが変わらないようにして
> ください。`CAMERA_HOST` に書くのはカメラのLAN IPであり、Tailscaleの `100.x` IP
> ではありません。

## 3. カメラへのアクセスをこのPCだけに絞る

`/32` は配布する経路をカメラ1台に限定しますが、アクセスできるtailnet端末までは
限定しません。Tailscale管理画面の **Access controls** で、既存ポリシーに次のGrantを
追加してください。`<FAMILIAR_PC_TAILSCALE_IP>` は、このPCの `tailscale ip -4` の
結果へ置き換えます。

```json
{
  "src": ["<FAMILIAR_PC_TAILSCALE_IP>/32"],
  "dst": ["<CAMERA_LAN_IP>/32"],
  "ip": ["tcp:554", "tcp:2020", "icmp:*"]
}
```

これは映像のRTSP、首振りのONVIF、疎通確認のpingだけを許可する例です。カメラ音声を
有効にする場合は、実際に使う追加ポートだけを同じGrantへ加えてください。

> TailscaleのGrantは加算式です。既定のallow-allなど、同じカメラへ到達できる広い
> Grantが残っていると、この狭いGrantでは打ち消せません。既存ルールも合わせて
> 狭めてください。

## 4. 複数カメラを登録する

初回起動時、既存の `.env` とリポジトリ直下の `.env.camera-*` は自動的に
`~/.familiar_ai/cameras.json` へ移行されます。以後はこのJSONがカメラ設定の正本です。
保存時のパーミッションは `0600` になります。

手動で作る場合は [サンプル](../config/cameras.json.example) をコピーし、次のように
設定します。

```json
{
  "version": 1,
  "default_camera": "main",
  "cameras": [
    {
      "id": "main",
      "label": "普段のカメラ",
      "host": "192.168.1.100",
      "username": "camera-user",
      "password": "camera-password",
      "onvif_port": 2020,
      "go2rtc_stream": "tapo_cam",
      "connection": "warm"
    },
    {
      "id": "travel",
      "label": "外出用カメラ",
      "host": "<CAMERA_LAN_IP>",
      "username": "camera-user",
      "password": "camera-password",
      "onvif_port": 2020,
      "go2rtc_stream": "tapo_cam_remote",
      "connection": "on_demand"
    }
  ]
}
```

- `default_camera`: 従来の `see()` / `look()` が使うカメラID。
- `connection: "warm"`: 起動中ずっと映像を待機。既定カメラ向け。
- `connection: "on_demand"`: `see_camera` の実行時だけ接続。外出用・予備向け。
- `tapo_password` / `tapo_hash`: カメラスピーカーを使う場合だけ設定。
- `FAMILIAR_CAMERAS_CONFIG`: JSONを別の場所へ置く場合のパス指定。

3台目以降も `cameras` 配列へ同じ形式で追加できます。`id` と
`go2rtc_stream` はカメラごとに一意にしてください。

## 5. 疎通確認と起動

このPCでTailnetへログインした後に確認します。

```bash
tailscale status
ping -c 3 <CAMERA_LAN_IP>
timeout 3 bash -c '</dev/tcp/<CAMERA_LAN_IP>/554' && echo 'RTSP OK'
timeout 3 bash -c '</dev/tcp/<CAMERA_LAN_IP>/2020' && echo 'ONVIF OK'
./run.sh --gui
```

起動後は通常の会話で既定カメラを使えるほか、モデルが次のツールを選べます。

- `see()` / `look()`: `default_camera` の映像・PTZ。
- `see_camera(camera="travel")`: 指定カメラをその場で接続して撮影。
- `look_camera(camera="travel", direction="left")`: 指定カメラのPTZ。

従来の一時的なdotenv切り替えも互換機能として残っています。

```bash
FAMILIAR_CAMERA_PROFILE=/path/to/camera.env ./run-remote-camera.sh --gui
./run.sh --camera-profile /path/to/camera.env --gui
```

`--camera-profile` はカメラ・go2rtc・カメラ音声に関係する設定だけを一時的に切り替え、
その起動では複数カメラJSONを使いません。
プロファイルにないカメラ設定は1台目から引き継がず、既定値へ戻ります。一方、
APIキー、モデル、人格、記憶、自律性などは既存 `.env` の設定を維持します。
プロファイル起動中は、1台目の `.env` を誤って上書きしないようGUIの設定保存を
停止します。2台目の値を変えるときは `.env.camera-remote` を直接編集してください。

JSON内の全 `go2rtc_stream` は既存のgo2rtc設定を消さずに同期されます。登録しただけでは
カメラへ接続せず、映像取得やカメラスピーカー再生でそのストリームを使ったときだけ
接続します。このため、普段オフラインの外出用カメラを登録したままでも、未使用中に
接続エラーを繰り返すことはありません。

## 切り分け

- `ping` から失敗する: subnet routeの承認、カメラ側端末の常時起動、カメラIPを確認。
- LinuxのこのPCだけ到達しない: `sudo tailscale set --accept-routes` を再確認。
- RTSPだけ失敗する: TapoアプリのカメラアカウントとTCP 554を確認。
- 首振りだけ失敗する: `CAMERA_ONVIF_PORT`（Tapo C220の一般的な値は2020）を確認。
- 1台目と2台目のLAN IPが同じ: `/32` routeでも意図しない宛先へ向く場合があるため、
  2台目側LANのアドレス帯を変更するのが確実です。
