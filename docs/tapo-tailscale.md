# TapoカメラをTailscale越しに使う

既存の `.env` と1台目のカメラ設定は変更しません。2台目を使うときだけ、
`.env.camera-remote` を重ねて起動します。

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

## 4. 2台目の設定を記入する

リポジトリ直下の `.env.camera-remote` に、2台目Tapoの値だけを書きます。
このファイルは `.gitignore` 対象です。

```dotenv
CAMERA_HOST=<CAMERA_LAN_IP>
CAMERA_USERNAME=<TAPOアプリで作ったカメラローカルユーザー>
CAMERA_PASSWORD="<カメラローカルパスワード>"
CAMERA_ONVIF_PORT=2020
```

## 5. 疎通確認と起動

このPCでTailnetへログインした後に確認します。

```bash
tailscale status
ping -c 3 <CAMERA_LAN_IP>
timeout 3 bash -c '</dev/tcp/<CAMERA_LAN_IP>/554' && echo 'RTSP OK'
timeout 3 bash -c '</dev/tcp/<CAMERA_LAN_IP>/2020' && echo 'ONVIF OK'
./run-remote-camera.sh --gui
```

通常の `./run.sh` / `./run-gui.sh` は引き続き既存 `.env` の1台目を使います。
2台目を使う場合だけ `./run-remote-camera.sh` を使ってください。

別名のプロファイルを試す場合は、次のどちらでも指定できます。

```bash
FAMILIAR_CAMERA_PROFILE=/path/to/camera.env ./run-remote-camera.sh --gui
./run.sh --camera-profile /path/to/camera.env --gui
```

`--camera-profile` はカメラ・go2rtc・カメラ音声に関係する設定だけを切り替えます。
プロファイルにないカメラ設定は1台目から引き継がず、既定値へ戻ります。一方、
APIキー、モデル、人格、記憶、自律性などは既存 `.env` の設定を維持します。
プロファイル起動中は、1台目の `.env` を誤って上書きしないようGUIの設定保存を
停止します。2台目の値を変えるときは `.env.camera-remote` を直接編集してください。

## 切り分け

- `ping` から失敗する: subnet routeの承認、カメラ側端末の常時起動、カメラIPを確認。
- LinuxのこのPCだけ到達しない: `sudo tailscale set --accept-routes` を再確認。
- RTSPだけ失敗する: TapoアプリのカメラアカウントとTCP 554を確認。
- 首振りだけ失敗する: `CAMERA_ONVIF_PORT`（Tapo C220の一般的な値は2020）を確認。
- 1台目と2台目のLAN IPが同じ: `/32` routeでも意図しない宛先へ向く場合があるため、
  2台目側LANのアドレス帯を変更するのが確実です。
