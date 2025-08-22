import cv2
import tempfile
import numpy as np
import os


def read_video(video_path, max_frames=None, skip_frames=1, resize_width=None):
    """Read video frames with optional skipping and resizing for efficiency."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    if not cap.isOpened():
        raise IOError(f"❌ Could not open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only keep every nth frame
        if frame_count % skip_frames == 0:
            if resize_width is not None:
                h, w = frame.shape[:2]
                aspect_ratio = h / w
                target_height = int(resize_width * aspect_ratio)
                frame = cv2.resize(frame, (resize_width, target_height))
            frames.append(frame)

        frame_count += 1

        # Stop if we reached max_frames
        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def frames_to_video_bytes(frames, fps=30):
    """Convert a list of frames to MP4 video bytes using a temporary file."""
    if not frames:
        return None

    height, width = frames[0].shape[:2]

    # Create temporary file
    tmp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file_path = tmp_file.name

        # Try H.264, fallback to mp4v
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(tmp_file_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_file_path, fourcc, fps, (width, height))

        # Write frames
        for frame in frames:
            if frame.dtype != "uint8":
                frame = frame.astype("uint8")
            out.write(frame)

        out.release()

        if not os.path.exists(tmp_file_path) or os.path.getsize(tmp_file_path) == 0:
            return None

        with open(tmp_file_path, "rb") as f:
            video_bytes = f.read()

        return video_bytes

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


def process_video(
    video_file,
    tracker,
    team_assigner,
    player_ball_assigner,
    max_frames=300,
    skip_frames=2,
    resize_width=640,
    fast_mode=True,
):
    """Process a video using cached models and optimizations."""
    import tempfile, os, numpy as np

    tmp_file_path = None
    try:
        # Save uploaded file to temp
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        # Read frames
        video_frames = read_video(
            tmp_file_path,
            max_frames=max_frames if fast_mode else None,
            skip_frames=skip_frames if fast_mode else 1,
            resize_width=resize_width if fast_mode else None,
        )

        if not video_frames:
            raise ValueError("No frames could be read from the video file")

        print(f"Processing {len(video_frames)} frames...")

        # Get object tracks
        tracks = tracker.get_object_tracks(video_frames)

        # Interpolate missing ball positions
        ball_tracks = tracker.interpolate_ball_positions(tracks["ball"])
        tracks["ball"] = ball_tracks

        # Reset team assigner state
        team_assigner.team_colors = {}
        team_assigner.player_team_dict = {}

        # ✅ Adapt player-ball distance threshold by resolution
        frame_h, frame_w = video_frames[0].shape[:2]
        player_ball_assigner.max_player_ball_distance = max(50, int(frame_w * 0.05))
        print(f"Using max_player_ball_distance={player_ball_assigner.max_player_ball_distance}")

        # ---------------- Player + Team Processing ----------------
        if "players" in tracks and tracks["players"]:
            first_players = tracks["players"][0]

            # Assign team colors if players exist
            if first_players:
                team_assigner.assign_team_color(video_frames[0], first_players)

                frame_step = 3 if fast_mode else 1

                # Assign teams to players in sampled frames
                for frame_num in range(0, len(tracks["players"]), frame_step):
                    player_track = tracks["players"][frame_num]
                    for player_id, track in player_track.items():
                        team = team_assigner.get_player_team(
                            video_frames[frame_num], track["bbox"], player_id
                        )
                        tracks["players"][frame_num][player_id]["team"] = team
                        tracks["players"][frame_num][player_id]["team_color"] = (
                            team_assigner.team_colors[team]
                        )

                # Fill skipped frames in fast mode
                if fast_mode and frame_step > 1:
                    for frame_num, player_track in enumerate(tracks["players"]):
                        for player_id, track in player_track.items():
                            if (
                                "team" not in track
                                and player_id in team_assigner.player_team_dict
                            ):
                                team = team_assigner.player_team_dict[player_id]
                                tracks["players"][frame_num][player_id]["team"] = team
                                tracks["players"][frame_num][player_id]["team_color"] = (
                                    team_assigner.team_colors[team]
                                )

            # ---------------- Ball Control Processing ----------------
            team_ball_control = []
            for frame_num, player_track in enumerate(tracks["players"]):
                ball_data = tracks["ball"][frame_num]
                if not ball_data or 1 not in ball_data:
                    # No ball detected → carry over last possession
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
                    continue

                ball_bbox = ball_data[1]["bbox"]
                assigned_player = player_ball_assigner.assign_ball_to_player(
                    player_track, ball_bbox
                )

                if assigned_player != -1:
                    tracks["players"][frame_num][assigned_player]["has_ball"] = True
                    player_team = tracks["players"][frame_num][assigned_player].get("team")

                    if player_team is None:
                        # If missing → backfill from dictionary
                        player_team = team_assigner.player_team_dict.get(assigned_player, 1)

                    team_ball_control.append(player_team)
                else:
                    # No player close enough → carry last team
                    team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)

        else:
            # No players detected → default control to team 1
            team_ball_control = [1] * len(video_frames)

        # ---------------- Sync Ball Control Length ----------------
        if len(team_ball_control) < len(video_frames):
            last_team = team_ball_control[-1] if team_ball_control else 1
            team_ball_control.extend([last_team] * (len(video_frames) - len(team_ball_control)))
        elif len(team_ball_control) > len(video_frames):
            team_ball_control = team_ball_control[:len(video_frames)]

        team_ball_control = np.array(team_ball_control)

        # ---------------- Draw Annotations ----------------
        print("Drawing annotations...")
        output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

        return output_video_frames, tracks

    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
