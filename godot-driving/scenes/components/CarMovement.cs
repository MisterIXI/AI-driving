using Godot;
using System;

public partial class CarMovement : StaticBody3D
{
	[Export] private RayCast3D _rayFront;
	[Export] private RayCast3D _rayBack;
	[Export] private RayCast3D _rayMiddle;
	[Export(PropertyHint.Range, "0.1,10")] private float _carHeight = 6.5f;
	[Export(PropertyHint.Range, "0.1,10")] private float _carSpeed = 1.0f;

	public override void _Process(double delta)
	{



		// orient cars to ground with raycasts
		if (_rayBack.IsColliding() && _rayFront.IsColliding())
		{
			Vector3 frontPos = _rayFront.GetCollisionPoint();
			Vector3 backPos = _rayBack.GetCollisionPoint();
			Vector3 direction = frontPos - backPos;
			Vector3 middleHit = _rayMiddle.GetCollisionPoint();
			// Normalize the direction vector
			LookAt(Position + direction, Vector3.Up);
			// Position += Vector3.Up * 6.5f;
			Position = middleHit + Vector3.Up * _carHeight;
			GD.Print("middleHit: " + middleHit);
		}
		else
		{
			// Position += Vector3.Down * 0.5f * (float)delta;
		}
		// move cars forward
		Vector3 directionForward = Transform.Basis.Z * _carSpeed;
		Position += new Vector3(directionForward.X, 0, 0);
		// Position = (Transform.Basis.Z * 5 * (float)delta).Slide(Vector3.Left);
		_rayBack.Position = Position + Transform.Basis.Z * 0.5f;
		_rayFront.Position = Position + Transform.Basis.Z * -0.5f;
		_rayMiddle.Position = Position;

	}
}
