from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'projects/AlgEngine/scripts/diffusiondrive'))
from innovation3.image_transport import ImageBuffer, metadata_from_wire, metadata_to_wire


class VisualTransportTest(unittest.TestCase):
    def test_metadata_roundtrip_keeps_arrays_and_mapping(self):
        source = {'cams': {'front': np.arange(6, dtype=np.float32).reshape(2, 3)},
                  'pose': np.eye(4, dtype=np.float64), 'token': 'frame'}
        self.assertEqual(metadata_from_wire(metadata_to_wire(source))['token'], 'frame')
        np.testing.assert_array_equal(metadata_from_wire(metadata_to_wire(source))['cams']['front'], source['cams']['front'])

    def test_shared_buffer_checks_identity_and_checksum(self):
        owner = ImageBuffer()
        reader = ImageBuffer(owner.fd)
        identity = {'scene': 's', 'step': 2, 'state_hash': 'h', 'token': 't'}
        try:
            descriptor = owner.publish({'front': b'jpeg-bytes'}, identity)
            self.assertEqual(reader.receive(descriptor, identity)['front'], b'jpeg-bytes')
            with self.assertRaises(ValueError): reader.receive(descriptor, dict(identity, step=3))
            reader.release(identity)
            with self.assertRaises(ValueError): owner.release(dict(identity, step=3))
        finally:
            # reader and owner intentionally share the same fd; close only the
            # mapping in the reader before the owner closes the descriptor.
            reader.close()
            owner.close()

    def test_buffer_rejects_reuse_before_ack(self):
        owner = ImageBuffer()
        identity = {'scene': 's', 'step': 0, 'state_hash': 'h', 'token': 't'}
        try:
            owner.publish({'front': b'a'}, identity)
            with self.assertRaises(RuntimeError): owner.publish({'front': b'b'}, identity)
            owner.release(identity)
        finally:
            owner.close()


if __name__ == '__main__':
    unittest.main()
